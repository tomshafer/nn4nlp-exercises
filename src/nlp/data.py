"""Datasets."""

from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
import os

__any__ = ["build_vocabulary", "ClassesDataset"]


UNK_TOKEN = "<UNK>"


def build_vocabulary(texts: Iterable[List[str]]) -> Dict[str, int]:
    """A vocabulary is a dictionary that maps strings onto token IDs."""
    vocab = {UNK_TOKEN: 0}
    i = 1
    for line in texts:
        for word in line:
            if word not in vocab:
                vocab[word] = i
                i += 1
    print(f"Vocabulary size = {len(vocab)} tokens")
    return vocab


class _ClassesDataset(Dataset):
    """This is a simple torch Dataset over a text file."""

    def __init__(self, src: str, vocab: Optional[Dict] = None):
        self.docs = self.read_txt_file(src)
        self.vocab = vocab if vocab else build_vocabulary(x for x, _ in self.docs)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        x, y = self.docs[index]
        xvec = torch.zeros(len(self.vocab))
        for w in x:
            i = self.vocab[w] if w in self.vocab else self.vocab[UNK_TOKEN]
            xvec[i] += 1
        return xvec, y

    def __len__(self) -> int:
        return len(self.docs)

    @classmethod
    def read_txt_file(cls, src: str) -> List[List]:
        """Each row of 'src' is represented as a list of tokens."""
        output = []
        with open(src, "r") as f:
            for line in f:
                catg, text = line.split(" ||| ")
                x = [w.lower() for w in text.strip().split() if w]
                y = int(catg)
                output.append([x, y])
        return output


class ClassesDataset:
    """The ClassesDataset wraps the full example data."""

    def __init__(self, srcdir: str):
        self.train = _ClassesDataset(os.path.join(srcdir, "train.txt"))
        # NB: dev and test share train's vocab
        self.dev = _ClassesDataset(os.path.join(srcdir, "dev.txt"), self.train.vocab)
        self.test = _ClassesDataset(os.path.join(srcdir, "test.txt"), self.train.vocab)
