import random
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .utils import BaseDataset, Tokenizer


class ICDAR21(LightningDataModule):
    """
    Args:
        batch_size: The number of samples per batch.
        num_workers: The number of subprocesses to use for data loading.
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory
            before returning them.
    """

    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        dirname: str = None,
        train_file: str = None,
        val_file: str = None,
        test_file: str = None,
        vocab_file: str = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_dirname = dirname
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.vocab_file = vocab_file
        self.transform = {
            # TODO check these transforms
            "train": T.ToTensor(),
            "val/test": T.ToTensor(),
        }

    def setup(self, stage: Optional[str] = None) -> None:
        """Load images and latex_codes, and assign them to a `torch Dataset`.

        `self.train_dataset`, `self.val_dataset` and `self.test_dataset` will
        be assigned after this method is called.
        """
        self.tokenizer = Tokenizer.load(self.vocab_file)

        if stage in ("fit", None):
            self.train_dataset = BaseDataset(
                self.data_dirname,
                self.train_file,
                transform=self.transform["train"],
            )

            self.val_dataset = BaseDataset(
                self.data_dirname,
                self.val_file,
                transform=self.transform["val/test"],
            )

        if stage in ("test", None):
            self.test_dataset = BaseDataset(
                self.data_dirname,
                self.test_file,
                transform=self.transform["val/test"],
            )

    def collate_fn(self, batch):
        images, latex_codes = zip(*batch)
        B = len(images)
        max_length = max(len(latex_code) for latex_code in latex_codes)
        batched_indices = torch.zeros((B, max_length + 2), dtype=torch.long)
        for i in range(B):
            indices = self.tokenizer.encode(latex_codes[i])
            batched_indices[i, : len(indices)] = torch.tensor(indices, dtype=torch.long)
        return torch.stack(images), batched_indices

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
