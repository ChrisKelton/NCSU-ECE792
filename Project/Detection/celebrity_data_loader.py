__all__ = ["CelebrityDataFake", "CelebrityDataCFFN"]
import itertools
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Union, Optional
from typing import Tuple
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image


class CelebrityData(torch.utils.data.Dataset):
    def __init__(self, zip_file: Path, transform=None, seed: Optional[int] = None, use_tmpdir: bool = True):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.transform = transform
        self.use_tmpdir = use_tmpdir
        zipObj = None
        if self.use_tmpdir:
            self.tmpdirname = TemporaryDirectory()
            self.temp_real_dirname = Path(self.tmpdirname.name) / "real"
            with ZipFile(str(zip_file), 'r') as zipObj:
                # Extract all the contents of zip file to temp_dir
                zipObj.extractall(self.temp_real_dirname)
        elif zip_file.suffix in [".zip"]:
            with ZipFile(str(zip_file), 'r') as zipObj:
                zipObj.extractall()
        else:
            self.real_images = sorted(zip_file.glob("*.jpg"))

        if zipObj is not None:
            self.real_images = zipObj.namelist()[1:]

    def __getitem__(self, index):
        if self.use_tmpdir:
            img_path = Path(self.temp_real_dirname) / self.real_images[index]
        else:
            img_path = self.real_images[index]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.real_images)

    def cleanup(self):
        if self.use_tmpdir:
            print("Cleaning up zip files")
            self.tmpdirname.cleanup()


class CelebrityDataFake(CelebrityData):

    def __init__(self, base_path: Path, transform=None, seed=None):
        '''
        Folder structure of base_path should be
        base_path -> RealFaces/FakeFaces
        RealFaces -> .zip
        FakeFaces -> GANType -> .zip (e.g., FakeFaces -> PGGAN -> .zip)
        '''
        self.real_images_path = Path(base_path) / "RealFaces"
        real_images_zip_files = sorted(self.real_images_path.glob("*.zip"))
        if len(real_images_zip_files) != 1:
            raise RuntimeError(
                f"Got more than or less than 1 zip file in '{self.real_images_path}'. Got '{len(real_images_zip_files)}'")
        self.real_images_zip_file = real_images_zip_files[0]
        super().__init__(self.real_images_zip_file, transform, seed)

        self.fake_images_path = Path(base_path) / "FakeFaces"
        self.fake_image_gan_names = os.listdir(str(self.fake_images_path))
        self.fake_images: Dict[str, List[Union[str, Path]]] = {}
        for fake_image_gan_name in self.fake_image_gan_names:
            fake_image_gan_path = self.fake_images_path / fake_image_gan_name
            zip_file = sorted(fake_image_gan_path.glob("*.zip"))
            temp_gan_dirname = Path(self.tmpdirname.name) / fake_image_gan_name
            if len(zip_file) == 1:
                zip_file = zip_file[0]
                with ZipFile(str(zip_file), 'r') as zipObj:
                    zipObj.extractall(temp_gan_dirname)
                self.fake_images.update({fake_image_gan_name: zipObj.namelist()})
            else:
                fake_image_paths = sorted(fake_image_gan_path.glob("*.jpg"))
                self.fake_images.update({fake_image_gan_name: fake_image_paths})

        self.n_fake_gans = len(list(self.fake_images.keys()))
        # according to Deep Fake Image Detection Based on Pairwise Learning, we need to make combinations for all
        # real images with all fake images
        fake_img_list = []
        for fake_imgs in self.fake_images.values():
            fake_img_list.extend(fake_imgs)
        self.fake_img_list = fake_img_list

        self.real_images = self.real_images[:self.len_of_fake_images + 1]

    def fake_image_rand_selection(self, index):
        rand_selection = self.rng.uniform(low=-0.499, high=len(self.fake_images) - 0.501)
        gan_selection = self.fake_image_gan_names[int(np.round(rand_selection))]

        return self.fake_images.get(gan_selection)[index // len(self.fake_images)]

    @property
    def len_of_fake_images(self) -> int:
        total_len = 0
        for val in self.fake_images.values():
            total_len += len(val)

        return total_len

    def len_of_real_and_fake(self):
        return len(self.real_images) + self.len_of_fake_images

    def __len__(self):
        return len(self.image_combos)


def number_of_combinations(n_objs: int, r_at_a_time: int) -> int:
    num = math.factorial(n_objs)
    den = math.factorial(n_objs - r_at_a_time) * math.factorial(r_at_a_time)
    return int(num / den)


def get_n_objs_for_a_number_of_combinations_with_2_at_a_time(combs: int) -> int:
    return int((1 + math.sqrt(1 + (4*combs * 2))) / 2)


class CelebrityDataCFFN(CelebrityDataFake):

    def __init__(self, base_path: Path, transform=None, seed=None, n_combinations: int = 2e6):
        super().__init__(base_path, transform, seed)

        # for training the CFFN we want fake-fake pairs & real-real pairs
        self.n_imgs_for_combinations = get_n_objs_for_a_number_of_combinations_with_2_at_a_time(n_combinations)

        # only making combinations between images made by the same GAN
        # could try experimenting with combinations of images between different GANs
        self.fake_image_combos: Dict[str, list] = {}
        for gan_name, img_list in self.fake_images.items():
            self.fake_image_combos.update({gan_name: list(itertools.combinations(img_list[:self.n_imgs_for_combinations], 2))})

        self.real_image_combos = list(itertools.combinations(self.real_images[:self.n_imgs_for_combinations], 2))

    def __getitem__(self, index):
        img0_path, img1_path, pair_indicator = self.choose_real_or_fake_pair(index)
        img0 = Image.open(img0_path).convert('RGB')
        if self.transform is not None:
            img0 = self.transform(img0)

        img1 = Image.open(img1_path).convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)

        return img0, img1, pair_indicator

    # def choose_real_or_fake_pair(self, index) -> Tuple[str, str, int]:
    #     if self.rng.standard_normal() > 0:
    #         img_pair = next(itertools.islice(self.real_image_combos, index, None))
    #         pair_indicator = 1
    #     else:
    #         img_pair = self.fake_image_rand_selection(index)
    #         pair_indicator = 0
    #
    #     return img_pair[0], img_pair[1], pair_indicator

    def choose_real_or_fake_pair(self, index) -> Tuple[str, str, int]:
        if self.rng.standard_normal() > 0:
            img_pair = self.real_image_combos[index]
            pair_indicator = 1
        else:
            img_pair = self.fake_image_rand_selection(index)
            pair_indicator = 0

        return img_pair[0], img_pair[1], pair_indicator

    # def fake_image_rand_selection(self, index) -> Tuple[str, str]:
    #     rand_selection = self.rng.uniform(low=-0.499, high=len(self.fake_images) - 0.501)
    #     gan_selection = self.fake_image_gan_names[int(np.round(rand_selection))]
    #
    #     iter_combo = self.fake_image_combos.get(gan_selection)
    #     return next(itertools.islice(iter_combo, index // len(self.fake_image_gan_names), None))

    def fake_image_rand_selection(self, index) -> Tuple[str, str]:
        rand_selection = self.rng.uniform(low=-0.499, high=self.n_fake_gans - 0.501)
        gan_selection = self.fake_image_gan_names[int(np.round(rand_selection))]

        return self.fake_image_combos[gan_selection][index]

    def __len__(self):
        return len(self.real_image_combos)