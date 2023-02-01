__all__ = ["JsonSerial"]
import jsonpickle
from typing import Union
from pathlib import Path
from files import get_real_path


class JsonSerial:
    def to_json_string(self) -> str:
        return jsonpickle.dumps(self, indent=2)

    def save_json(self, path: Union[str, Path]):
        json_str = self.to_json_string()
        Path(get_real_path(path)).write_text(json_str)

    @classmethod
    def from_json_string(cls, json_str: str):
        obj = jsonpickle.loads(json_str)
        return obj

    @classmethod
    def read_json(cls, path: Union[str, Path]):
        json_str = Path(get_real_path(path)).read_text()
        return cls.from_json_string(json_str)
