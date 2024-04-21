import os
import json
from abc import abstractmethod

import pandas as pd
from loguru import logger

from ..base import BaseDataset


class InspiredDataset(BaseDataset):

    def __init__(self, opts, tokenizer, restore=False, datafolder_path="data/inspired"):
        super().__init__(opts, tokenizer, restore, datafolder_path)

    def _load_dataset(self):
        """Load the raw Redial dataset"""

        train_data = self._load_raw_data("raw_inspired_dataset/train.tsv")
        valid_data = self._load_raw_data("raw_inspired_dataset/dev.tsv")
        test_data = self._load_raw_data("raw_inspired_dataset/test.tsv")

        return train_data, valid_data, test_data

    def _load_raw_data(self, file_name):
        """Load the raw inspired dataset """
        file_path = os.path.join(self.datafolder_path, file_name)
        data = pd.read_csv(file_path, sep="\t")

        logger.debug(f"successfully load {len(data)} raw inspired data from {file_name}")

        return data

    @abstractmethod
    def _preprocess(self, train, valid, test):
        """Preprocessing the raw Redial dataset"""
        pass
