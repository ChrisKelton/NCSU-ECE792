__all__ = ["DatasetGenerator"]
import random
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from PIL.Image import open as PILOpen
from joblib import Parallel, delayed

image_fun: Callable = lambda x: np.array(PILOpen(x))


class DatasetGenerator:

    def __init__(self, base_dir_path: Path, batch_size: int, shuffle: bool = True, n_parallel_jobs: int = 4, seed: Optional[int] = None):
        self.image_paths = sorted(base_dir_path.glob("*.jpg"))
        if shuffle:
            if seed is not None:
                random.Random(seed).shuffle(self.image_paths)
            else:
                random.shuffle(self.image_paths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_parallel_jobs = n_parallel_jobs
        self.seed = seed
        self.index = 0

    def on_epoch_end(self):
        if self.shuffle:
            if self.seed is not None:
                random.Random(self.seed).shuffle(self.image_paths)
            else:
                random.shuffle(self.image_paths)

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        batch = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__get_data(batch)

    def __get_data(self, batch):
        images = Parallel(n_jobs=self.n_parallel_jobs)(delayed(image_fun)(f) for f in batch)
        return np.array(images)

    def __next__(self):
        return self.next()

    def next(self):
        x = self.__getitem__(self.index)
        if self.index + 1 >= self.__len__():
            self.index = 0
            self.on_epoch_end()
        else:
            self.index += 1

        return x

    def __iter__(self):
        for _ in range(self.__len__()):
            yield self.__next__()

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration

    def close(self):
        """Raise GeneratorExit inside generator."""
        try:
            self.throw(GeneratorExit)
        except (GeneratorExit, StopIteration):
            pass
        else:
            raise RuntimeError("generator ignored GeneratorExit")