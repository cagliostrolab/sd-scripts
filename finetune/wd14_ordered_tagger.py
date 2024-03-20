import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

import library.train_util as train_util

IMAGE_SIZE = 448
DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
FILES_ONNX = ["model.onnx"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = FILES[-1]

GENDER_TAGS = [
    "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple girls",
    "1boy", "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple boys", "male focus",
    "1other", "2others", "3others", "4others", "5others", "6+others", "multiple others", "other focus",
]


def preprocess_image(image):
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)

    image = image.astype(np.float32)
    return image


class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])

        try:
            image = Image.open(img_path).convert("RGB")
            image = preprocess_image(image)
            tensor = torch.tensor(image)
        except Exception as e:
            print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (tensor, img_path)


def collate_fn_remove_corrupted(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def main(args):
    if not os.path.exists(args.model_dir) or args.force_download:
        print(f"downloading wd14 tagger model from hf_hub. id: {args.repo_id}")
        files = FILES
        if args.onnx:
            files += FILES_ONNX
        for file in files:
            hf_hub_download(args.repo_id, file, cache_dir=args.model_dir, force_download=True, force_filename=file)
        for file in SUB_DIR_FILES:
            hf_hub_download(
                args.repo_id,
                file,
                subfolder=SUB_DIR,
                cache_dir=os.path.join(args.model_dir, SUB_DIR),
                force_download=True,
                force_filename=file,
            )
    else:
        print("using existing wd14 tagger model")

    if args.onnx:
        import onnx
        import onnxruntime as ort

        onnx_path = f"{args.model_dir}/model.onnx"
        print("Running wd14 tagger with onnx")
        print(f"loading onnx model: {onnx_path}")

        if not os.path.exists(onnx_path):
            raise Exception(
                f"onnx model not found: {onnx_path}, please redownload the model with --force_download"
                + " / onnxモデルが見つかりませんでした。--force_downloadで再ダウンロードしてください"
            )

        model = onnx.load(onnx_path)
        input_name = model.graph.input[0].name
        try:
            batch_size = model.graph.input[0].type.tensor_type.shape.dim[0].dim_value
        except:
            batch_size = model.graph.input[0].type.tensor_type.shape.dim[0].dim_param

        if args.batch_size != batch_size and type(batch_size) != str:
            print(
                f"Batch size {args.batch_size} doesn't match onnx model batch size {batch_size}, use model batch size {batch_size}"
            )
            args.batch_size = batch_size

        del model

        ort_sess = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"],
        )
    else:
        from tensorflow.keras.models import load_model

        model = load_model(f"{args.model_dir}")

    with open(os.path.join(args.model_dir, CSV_FILE), "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        l = [row for row in reader]
        header = l[0]  # tag_id,name,category,count
        rows = l[1:]
    assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category", f"unexpected csv format: {header}"

    general_tags = [row[1] for row in rows[1:] if row[2] == "0"]
    character_tags = [row[1] for row in rows[1:] if row[2] == "4"]

    train_data_dir_path = Path(args.train_data_dir)
    image_paths = train_util.glob_images_pathlib(train_data_dir_path, args.recursive)
    print(f"found {len(image_paths)} images.")

    tag_freq = {}

    caption_separator = args.caption_separator
    stripped_caption_separator = caption_separator.strip()
    undesired_tags = set(args.undesired_tags.split(stripped_caption_separator))

    def run_batch(path_imgs):
        imgs = np.array([im for _, im in path_imgs])
    
        if args.onnx:
            if len(imgs) < args.batch_size:
                imgs = np.concatenate([imgs, np.zeros((args.batch_size - len(imgs), IMAGE_SIZE, IMAGE_SIZE, 3))], axis=0)
            probs = ort_sess.run(None, {input_name: imgs})[0]  # onnx output numpy
            probs = probs[: len(path_imgs)]
        else:
            probs = model(imgs, training=False)
            probs = probs.numpy()
    
        for (image_path, _), prob in zip(path_imgs, probs):
            combined_tags = []
            raw_tags = []
            gender_tags_detected = []
            character_tags_detected = []
            series_tags_detected = [args.series_tag] if args.series_tag else []
            other_general_tags = []
    
            for i, p in enumerate(prob[4:]):  # Skip the first 4 as they are ratings
                tag_name = general_tags[i] if i < len(general_tags) else character_tags[i - len(general_tags)]
                if args.remove_underscore and len(tag_name) > 3:  # ignore emoji tags
                    tag_name = tag_name.replace("_", " ")
    
                if tag_name in undesired_tags:
                    continue
    
                if tag_name in GENDER_TAGS and p >= args.general_threshold:
                    gender_tags_detected.append(tag_name)
                elif i >= len(general_tags) and p >= args.character_threshold:
                    character_tags_detected.append(tag_name)
                elif p >= args.general_threshold:
                    other_general_tags.append(tag_name)
    
                if p >= args.general_threshold or (i >= len(general_tags) and p >= args.character_threshold):
                    tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                    raw_tags.append(tag_name)
    
            # Ordering logic
            if args.tags_ordering:
                # Sort gender tags based on their predefined list order
                gender_tags_detected.sort(key=lambda x: GENDER_TAGS.index(x))
                
                # Add manually specified character tags if not already detected
                manual_character_tags = args.character_tag.split(args.caption_separator) if args.character_tag else []
                for tag in manual_character_tags:
                    if tag not in character_tags_detected:
                        character_tags_detected.append(tag)
    
                combined_tags = gender_tags_detected + character_tags_detected + series_tags_detected
                if other_general_tags:
                    tags_text = args.caption_separator.join(combined_tags) + args.caption_separator + args.tags_separator + " " + args.caption_separator.join(other_general_tags)
                else:
                    tags_text = args.caption_separator.join(combined_tags)
            else:
                tags_text = args.caption_separator.join(raw_tags)
    
            # Handling existing captions based on the append_tags flag
            caption_file = os.path.splitext(image_path)[0] + args.caption_extension
            if args.append_tags:
                if os.path.exists(caption_file):
                    with open(caption_file, "rt", encoding="utf-8") as f:
                        existing_content = f.read().strip("\n")
                    existing_tags = [tag.strip() for tag in existing_content.split(stripped_caption_separator) if tag.strip()]
                    new_tags = [tag for tag in combined_tags if tag not in existing_tags]
                    ordered_tags_text = args.caption_separator.join(existing_tags + new_tags)
    
            with open(caption_file, "wt", encoding="utf-8") as f:
                f.write(tags_text)
                if args.debug:
                    print(f"\n{image_path}:\n  Tags: {tags_text}")

    if args.max_data_loader_n_workers is not None:
        dataset = ImageLoadingPrepDataset(image_paths)
        data = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.max_data_loader_n_workers,
            collate_fn=collate_fn_remove_corrupted,
            drop_last=False,
        )
    else:
        data = [[(None, ip)] for ip in image_paths]

    b_imgs = []
    for data_entry in tqdm(data, smoothing=0.0):
        for data in data_entry:
            if data is None:
                continue

            image, image_path = data
            if image is not None:
                image = image.detach().numpy()
            else:
                try:
                    image = Image.open(image_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image = preprocess_image(image)
                except Exception as e:
                    print(f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}")
                    continue
            b_imgs.append((image_path, image))

            if len(b_imgs) >= args.batch_size:
                b_imgs = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
                run_batch(b_imgs)
                b_imgs.clear()

    if len(b_imgs) > 0:
        b_imgs = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
        run_batch(b_imgs)

    if args.frequency_tags:
        sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
        print("\nTag frequencies:")
        for tag, freq in sorted_tags:
            print(f"{tag}: {freq}")

    print("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_WD14_TAGGER_REPO, help="repo id for wd14 tagger on Hugging Face / Hugging Faceのwd14 taggerのリポジトリID")
    parser.add_argument("--model_dir", type=str, default="wd14_tagger_model", help="directory to store wd14 tagger model / wd14 taggerのモデルを格納するディレクトリ")
    parser.add_argument("--force_download", action="store_true", help="force downloading wd14 tagger models / wd14 taggerのモデルを再ダウンロードします")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument("--max_data_loader_n_workers", type=int, default=None, help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）")
    parser.add_argument("--caption_extension", type=str, default=".txt", help="extension of caption file / 出力されるキャプションファイルの拡張子")
    parser.add_argument("--thresh", type=float, default=0.35, help="threshold of confidence to add a tag / タグを追加するか判定する閾値")
    parser.add_argument("--general_threshold", type=float, default=None, help="threshold of confidence to add a tag for general category, same as --thresh if omitted / generalカテゴリのタグを追加するための確信度の閾値、省略時は --thresh と同じ")
    parser.add_argument("--character_threshold", type=float, default=None, help="threshold of confidence to add a tag for character category, same as --thresh if omitted / characterカテゴリのタグを追加するための確信度の閾値、省略時は --thresh と同じ")
    parser.add_argument("--recursive", action="store_true", help="search for images in subfolders recursively / サブフォルダを再帰的に検索する")
    parser.add_argument("--remove_underscore", action="store_true", help="replace underscores with spaces in the output tags / 出力されるタグのアンダースコアをスペースに置き換える")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--undesired_tags", type=str, default="", help="comma-separated list of undesired tags to remove from the output / 出力から除外したいタグのカンマ区切りのリスト")
    parser.add_argument("--frequency_tags", action="store_true", help="Show frequency of tags for images / 画像ごとのタグの出現頻度を表示する")
    parser.add_argument("--onnx", action="store_true", help="use onnx model for inference / onnxモデルを推論に使用する")
    parser.add_argument("--append_tags", action="store_true", help="Append captions instead of overwriting / 上書きではなくキャプションを追記する")
    parser.add_argument("--caption_separator", type=str, default=", ", help="Separator for captions, include space if needed / キャプションの区切り文字、必要ならスペースを含めてください")
    parser.add_argument("--tags_ordering", action="store_true", help="Order tags as gender, character, series, and general tags / タグを性別、キャラクター、シリーズ、一般タグの順に並べる")
    parser.add_argument("--character_tag", type=str, default="", help="Specific character tag to look for / 特定のキャラクタータグを探す")
    parser.add_argument("--series_tag", type=str, default="", help="Specific series tag to look for / 特定のシリーズタグを探す")
    parser.add_argument("--tags_separator", type=str, default="|||", help="Separator between different types of tags / 異なるタイプのタグ間の区切り文字")

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    if args.general_threshold is None:
        args.general_threshold = args.thresh
    if args.character_threshold is None:
        args.character_threshold = args.thresh

    main(args)
