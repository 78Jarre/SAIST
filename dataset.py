import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from tokenizer import tokenize


class SIRSTDataset(Dataset):
    def __init__(self, rows, training=False):
        (self.rows, self.training) = (rows, training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image"]) as f:
            image = f.convert("RGB")
        with Image.open(row["mask"]) as f:
            mask = f.convert("L")
        if image.size != mask.size:
            raise ValueError(f"Image/mask size mismatch: {row}")
        image = image.resize((1024, 1024), Image.Resampling.BILINEAR)
        mask = mask.resize((1024, 1024), Image.Resampling.NEAREST)
        if self.training:
            if random.random() < 0.5:
                (image, mask) = (
                    image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
                    mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
                )
            if random.random() < 0.5:
                (image, mask) = (
                    image.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
                    mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
                )
        image = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1)
        mask = torch.from_numpy((np.array(mask) > 0).astype(np.float32))[None]
        caption = tokenize(Path(row["caption"]).read_text(encoding="utf-8-sig").strip())[0]
        return dict(image=image, mask=mask, caption=caption)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


import hashlib

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def check_disjoint(*splits):
    (seen_paths, seen_hashes) = (set(), set())
    for rows in splits:
        paths = {row["image"] for row in rows}
        hashes = {hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest() for row in rows}
        if seen_paths & paths or seen_hashes & hashes:
            raise ValueError("Image overlap between splits (path or file content).")
        seen_paths.update(paths)
        seen_hashes.update(hashes)


def _file_index(directory, extensions):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result = {}
    for file in sorted(directory.iterdir()):
        if not file.is_file() or file.suffix.lower() not in extensions:
            continue
        target = file.resolve()
        if not target.is_relative_to(directory):
            raise ValueError(f"File escapes its dataset directory: {file}")
        if file.stem in result:
            raise ValueError(f"Ambiguous file stem in {directory}: {file.stem}")
        result[file.stem] = target
    return result


def _match_file(index, stem, suffixes, role):
    matches = [index[stem + suffix] for suffix in dict.fromkeys(suffixes) if stem + suffix in index]
    if len(matches) != 1:
        raise ValueError(
            f"{role} for {stem}: expected exactly one match for suffixes {suffixes}; found {matches}"
        )
    return matches[0]


def rows_from_ids(data_root, split_file, mask_suffix=None, caption_suffix=None):
    (data_root, split_file) = (Path(data_root).resolve(), Path(split_file).resolve())
    directories = {key: data_root / key for key in ("images", "masks", "descriptions")}
    for directory in directories.values():
        if not directory.resolve().is_relative_to(data_root):
            raise ValueError(f"Directory escapes data root: {directory}")
    images = _file_index(directories["images"], IMAGE_EXTENSIONS)
    masks = _file_index(directories["masks"], IMAGE_EXTENSIONS)
    captions = _file_index(directories["descriptions"], {".txt"})
    mask_suffixes = ["_pixels0", "", "_mask"] if mask_suffix is None else [mask_suffix]
    caption_suffixes = ["_description", ""] if caption_suffix is None else [caption_suffix]
    (result, seen) = ([], set())
    for (line_number, line) in enumerate(
        split_file.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        value = line.strip()
        if not value:
            continue
        if any((character in value for character in ("/", "\\", ":", "\x00"))) or value in (
            ".",
            "..",
        ):
            raise ValueError(
                f"{split_file}:{line_number}: expected an ID or image filename, not a path: {value!r}"
            )
        suffix = Path(value).suffix.lower()
        if suffix and suffix not in IMAGE_EXTENSIONS:
            raise ValueError(f"{split_file}:{line_number}: unsupported image suffix: {value!r}")
        stem = Path(value).stem if suffix else value
        if stem in seen:
            raise ValueError(f"{split_file}:{line_number}: duplicate ID {stem}")
        if stem not in images:
            raise FileNotFoundError(f"{split_file}:{line_number}: no image for {value!r}")
        image = images[stem]
        if suffix and image.suffix.lower() != suffix:
            raise FileNotFoundError(
                f"{split_file}:{line_number}: no image matching filename {value!r}"
            )
        mask = _match_file(masks, stem, mask_suffixes, "mask")
        caption = _match_file(captions, stem, caption_suffixes, "caption")
        if not caption.read_text(encoding="utf-8-sig").strip():
            raise ValueError(f"Empty caption: {caption}")
        seen.add(stem)
        result.append(dict(image=str(image), mask=str(mask), caption=str(caption)))
    if not result:
        raise ValueError(f"Empty split: {split_file}")
    return result


def train_pixel_stats(rows):
    import numpy as np
    from PIL import Image

    total = squared = 0.0
    count = 0
    for row in rows:
        with Image.open(row["image"]) as image:
            values = np.asarray(image.convert("RGB"), dtype=np.float64)
        total += values.sum()
        squared += np.square(values).sum()
        count += values.size
    if not count:
        raise ValueError("No training image pixels")
    mean = total / count
    std = max(0.0, squared / count - mean * mean) ** 0.5
    if std < 1e-06:
        raise ValueError("Training image standard deviation is zero")
    return ([mean], [std])
