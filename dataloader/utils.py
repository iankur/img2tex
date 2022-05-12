import os
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """A base Dataset class.

    Args:
        image_filenames: (N, *) feature vector.
        targets: (N, *) target vector relative to data.
        transform: Feature transformation.
        target_transform: Target transformation.
    """

    def __init__(
        self,
        root_dir: Path,
        filename: List[str],
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.samples = open(filename).read().strip().splitlines()
        self.transform = transform

    def __len__(self) -> int:
        """Returns the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Returns a sample from the dataset at the given index."""
        image_filename, latex_code = self.samples[idx].split(' ', 1)
        image_filepath = os.path.join(self.root_dir, image_filename)
        image = pil_loader(image_filepath, mode="L")
        if self.transform is not None:
            image = self.transform(image)
        # TODO latex_code required format may change depending on dataset
        return image, latex_code.strip().split()


def pil_loader(fp: Path, mode: str) -> Image.Image:
    with open(fp, "rb") as f:
        img = Image.open(f)
        return img.convert(mode)


class Tokenizer:
    def __init__(self, token_to_index: Optional[Dict[str, int]] = None) -> None:
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        assert token_to_index, "vocabulary with mapping from token to id?"
        self.token_to_index: Dict[str, int]
        self.index_to_token: Dict[int, str]

        self.token_to_index = token_to_index
        self.index_to_token = {index: token for token, index in self.token_to_index.items()}
        self.pad_index = self.token_to_index[self.pad_token]
        self.sos_index = self.token_to_index[self.sos_token]
        self.eos_index = self.token_to_index[self.eos_token]
        self.unk_index = self.token_to_index[self.unk_token]

        self.ignore_indices = {self.pad_index, self.sos_index, self.eos_index, self.unk_index}

    def __len__(self):
        return len(self.token_to_index)

    def encode(self, formula: List[str]) -> List[int]:
        indices = [self.sos_index]
        for token in formula:
            index = self.token_to_index.get(token, self.unk_index)
            indices.append(index)
        indices.append(self.eos_index)
        return indices

    def decode(self, indices: List[int], inference: bool = True) -> List[str]:
        tokens = []
        for index in indices:
            if index not in self.index_to_token:
                raise RuntimeError(f"Found an unknown index {index}")
            if index == self.eos_index:
                break
            if inference and index in self.ignore_indices:
                continue
            token = self.index_to_token[index]
            tokens.append(token)
        return tokens

    @classmethod
    def load(cls, filename: Union[Path, str]) -> "Tokenizer":
        """Create a `Tokenizer` from a mapping file outputted by `save`.

        Args:
            filename: Path to the file to read from.

        Returns:
            A `Tokenizer` object.
        """
        with open(filename) as f:
            token_to_index = json.load(f)
        return cls(token_to_index)
