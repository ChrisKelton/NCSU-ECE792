import shutil
import zipfile
from pathlib import Path
from typing import List, Union

import numpy as np
from tqdm import tqdm

from HW.HW1.mlp_mnist.files import get_real_path


def zip_files(files: List[Union[str, Path]], output_zip_path: Union[str, Path]):
    with zipfile.ZipFile(get_real_path(output_zip_path), mode="w") as archive:
        for file in files:
            archive.write(get_real_path(file))


def zip_files_into_multiple_zip_files(
    base_path: Union[str, Path],
    output_zip_base_path: Union[str, Path],
    remove_existing_outputs: bool = True,
    file_type: str = ".jpg",
    files_per_zip: int = 1000,
):
    all_files = sorted(Path(base_path).glob(f"*{file_type}"))
    number_of_zip_files = len(all_files) / files_per_zip
    if not number_of_zip_files.is_integer():
        number_of_zip_files = int(number_of_zip_files + 1)
    else:
        number_of_zip_files = int(number_of_zip_files)
    if Path(output_zip_base_path).exists() and remove_existing_outputs:
        shutil.rmtree(get_real_path(output_zip_base_path))
    Path(output_zip_base_path).mkdir(exist_ok=True, parents=True)
    print(f"Generating {number_of_zip_files} zip files with {files_per_zip} files per zip file at '{output_zip_base_path}'")
    for zip_file_idx in tqdm(range(number_of_zip_files)):
        beg_idx = zip_file_idx * files_per_zip
        end_idx = np.clip((zip_file_idx + 1) * files_per_zip, 0, len(all_files) - 1)
        files_to_zip = all_files[beg_idx:end_idx]
        zip_file_path = Path(output_zip_base_path) / f"{zip_file_idx}.zip"
        zip_files(files_to_zip, zip_file_path)


def main():
    # zip_file_path = get_real_path(Path("C:/Users/Chris/Documents/NCSU-Graduate/Courses/ECE792/Project/datasets/realimages/celeba/img_align_celeba"))
    # output_base_path = get_real_path(Path("C:/Users/Chris/Documents/NCSU-Graduate/Courses/ECE792/Project/datasets/realimages/celeba/img-align-celeba-zip-files"))
    zip_file_path = get_real_path(Path("C:/Users/Chris/Documents/NCSU-Graduate/Courses/ECE792/Project/datasets/realimages/celeba/processed_celeba_small/celeba/New Folder With Items"))
    output_base_path = get_real_path(Path("C:/Users/Chris/Documents/NCSU-Graduate/Courses/ECE792/Project/datasets/realimages/celeba/processed-celeba-small-zip-files"))
    zip_files_into_multiple_zip_files(zip_file_path, output_base_path, files_per_zip=100)


if __name__ == '__main__':
    main()