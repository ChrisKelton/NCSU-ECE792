from PIL import Image
from pathlib import Path
from typing import List
import numpy as np
from Project.utils.files import get_real_path
from tqdm import tqdm


def load_dataset_into_npy_file(img_dir_path: Path, suffix: List[str] = None, cnt_to_save: int = 100):
    if suffix is None:
        suffix = [".jpg"]
    img_paths = []
    img_dir_path = get_real_path(img_dir_path)
    for suffix_ in suffix:
        img_paths.extend(sorted(Path(img_dir_path).glob(f"*{suffix_}")))
    dataset_files = []
    for cnt, path in tqdm(enumerate(img_paths), total=len(img_paths)):
        if (cnt % cnt_to_save == 0 and cnt != 0) or (cnt == len(img_paths) - 1):
            numpy_imgset_path = Path(img_dir_path).parent / f"img_align_celeba--{cnt}.npy"
            np.save(get_real_path(numpy_imgset_path), np.array(dataset_files))
            dataset_files = []
        im = Image.open(get_real_path(path))
        im = np.array(im)
        dataset_files.append(im)


def aggregate_npy_files(numpy_imgset_paths: Path):
    numpy_file_paths = sorted(numpy_imgset_paths.glob("*.npy"))
    temp = []
    for np_file_path in numpy_file_paths:
        temp.append(np.load(get_real_path(np_file_path)))
    dataset = np.concatenate(temp, axis=0)
    dataset_output_path = numpy_imgset_paths / "dataset.npy"
    np.save(get_real_path(dataset_output_path), dataset)


def main():
    img_dir_path = Path("C:/Users/Chris/Documents/NCSU-Graduate/Courses/ECE792/Project/datasets/realimages/celeba/img_align_celeba")
    load_dataset_into_npy_file(img_dir_path, cnt_to_save=10000)
    aggregate_npy_files(img_dir_path.parent)


if __name__ == '__main__':
    main()
