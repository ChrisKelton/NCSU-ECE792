__all__ = ["load_mnist_dataset", "get_real_path"]
import os
import random
import subprocess
from pathlib import Path
from typing import List, Optional, Union

from mnist import MNIST


def command_line_execution(cmd: str, shell: bool = False) -> bool:
    try:
        with subprocess.Popen(cmd, stdin=subprocess.PIPE, shell=shell) as p:
            p.communicate()
    except Exception as e:
        print(f"Exception: '{e}'")
        return False

    return True


def get_real_path(path_: Union[str, Path]) -> Union[str, Path]:
    if os.name == "nt":
        return r"{}".format(os.path.realpath(path_).replace("/", "\\"))
    return Path(os.path.realpath(path_))


def unzip_gz_files(input: Path, overwrite_files: bool = False, shell: bool = False) -> bool:
    input = get_real_path(input)
    if Path(input).is_dir():
        input = [get_real_path(path_) for path_ in sorted(Path(input).glob("*.gz"))]
    else:
        input = [str(input)]

    cmd_ = "gzip -dk "
    results: List[bool] = []
    for path_ in input:
        if not Path(path_).with_suffix("").exists() or overwrite_files:
            print(f"Unzipping '{path_}'")
            if isinstance(path_, Path):
                path_ = str(path_)
            results.append(command_line_execution(cmd=r"{}".format(cmd_ + path_), shell=shell))

    return all(results)


def check_mnist_gz_files(mnist_dataset_dir: Path, overwrite_files: bool = False, shell: bool = False) -> bool:
    return unzip_gz_files(mnist_dataset_dir, overwrite_files=overwrite_files, shell=shell)


def load_mnist_dataset(mnist_dataset_dir: Path, overwrite_files: bool = False, shell: bool = False) -> Optional[MNIST]:
    mnist_dataset_dir = get_real_path(mnist_dataset_dir)
    if not check_mnist_gz_files(mnist_dataset_dir, overwrite_files=overwrite_files, shell=shell):
        print(f"GZIP execution not successfully implemented. Please unzip your .gz files and rerun application.")
        return None

    return MNIST(mnist_dataset_dir)


def main():
    mnist_data_sets_base_path = Path("../DATA/MNIST")
    mndata = load_mnist_dataset(mnist_data_sets_base_path)
    if mndata is None:
        raise RuntimeError(f"Failed to load in mnist dataset")
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    index = random.randrange(0, len(train_images))
    print("Example training image:\n")
    print(mndata.display(train_images[index]))
    print("\n")
    print(f"Example training label: '{train_labels[index]}'")
    print("\n\n\n")

    index = random.randrange(0, len(test_images))
    print("Example testing image:\n")
    print(mndata.display(test_images[index]))
    print("\n")
    print(f"Example testing label: '{test_labels[index]}'")


if __name__ == "__main__":
    main()
