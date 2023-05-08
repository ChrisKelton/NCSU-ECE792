__all__ = ["CelebrityData", "CelebrityDataCVAE"]
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CelebrityData(Dataset):
    def __init__(
        self,
        zip_file: Path,
        transform=None,
        seed: Optional[int] = None,
        use_tmpdir: bool = True,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        val_set: bool = False,
        test_set: bool = False,
    ):
        super().__init__()
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.val_set = val_set
        self.test_set = test_set
        self.rng = np.random.default_rng(seed)
        self.transform = transform
        self.use_tmpdir = use_tmpdir
        zipObj = None
        if self.use_tmpdir:
            self.tmpdirname = TemporaryDirectory()
            self.temp_real_dirname = Path(self.tmpdirname.name) / "real"
            with ZipFile(str(zip_file), 'r') as zipObj:
                zipObj.extractall(self.temp_real_dirname)
        elif zip_file.suffix in [".zip"]:
            with ZipFile(str(zip_file), 'r') as zipObj:
                zipObj.extractall()
        else:
            all_images = sorted(zip_file.glob("*.jpg"))

        if zipObj is not None:
            all_images = zipObj.namelist()[1:]

        n_imgs_for_validation_and_test = int(len(all_images) * (validation_ratio + test_ratio))
        end_train_idx = len(all_images) - n_imgs_for_validation_and_test
        self.real_images_train = all_images[:end_train_idx]
        end_val_idx = int(end_train_idx + (n_imgs_for_validation_and_test * validation_ratio))
        self.real_images_val = all_images[end_train_idx:end_val_idx]
        self.real_images_test = all_images[end_val_idx:]

    def __getitem__(self, index):
        if self.val_set:
            img_name = self.real_images_val[index]
        elif self.test_set:
            img_name = self.real_images_test[index]
        else:
            img_name = self.real_images_train[index]

        if self.use_tmpdir:
            img_path = Path(self.temp_real_dirname) / img_name
        else:
            img_path = img_name

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        if self.val_set:
            return len(self.real_images_val)
        elif self.test_set:
            return len(self.real_images_test)

        return len(self.real_images_train)

    def cleanup(self):
        if self.use_tmpdir:
            print("Cleaning up zip files")
            self.tmpdirname.cleanup()

    def update_dataset_type(self, val_set: bool = False, test_set: bool = False):
        if val_set:
            self.val_set = val_set
            self.test_set = False
        elif test_set:
            self.test_set = test_set
            self.val_set = False
        else:
            self.val_set = False
            self.test_set = False

class CelebrityDataCVAE(CelebrityData):
    def __init__(
            self,
            base_path: Path,
            attr_file: Path,
            img_transform=None,
            seed=None,
            use_tmpdir: bool = True,
            validation_ratio: float = 0.1,
            test_ratio: float = 0.1,
            val_set: bool = False,
            test_set: bool = False,
    ):
        super().__init__(
            base_path,
            img_transform,
            seed,
            use_tmpdir,
            validation_ratio,
            test_ratio,
            val_set,
            test_set,
        )
        self.attrs = pd.read_csv(str(attr_file), index_col=0)

    def __getitem__(self, index):
        if self.val_set:
            img_name = self.real_images_val[index]
        elif self.test_set:
            img_name = self.real_images_test[index]
        else:
            img_name = self.real_images_train[index]

        if self.use_tmpdir:
            img_path = self.temp_real_dirname / img_name
        else:
            img_path = img_name

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        attr = torch.Tensor(self.attrs.loc[int(Path(img_path).with_suffix("").name)])

        return img, attr