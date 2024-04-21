import json
import os
from abc import abstractmethod

from loguru import logger

from ..base import BaseDataset


class RedialDataset(BaseDataset):

    def __init__(self, opts, tokenizer, restore=False, datafolder_path="data/redial"):
        super().__init__(opts, tokenizer, restore, datafolder_path)

    def _load_dataset(self):
        """Load the raw Redial dataset"""

        train_data = self._load_raw_data("raw_redial_dataset/train_data.jsonl")
        valid_data = self._load_raw_data("raw_redial_dataset/valid_data.jsonl")
        test_data = self._load_raw_data("raw_redial_dataset/test_data.jsonl")

        return train_data, valid_data, test_data

    def _load_raw_data(self, file_name):
        """Load the raw redial dataset """
        data = []
        with open(os.path.join(self.datafolder_path, file_name)) as f:
            data.extend(json.loads(line) for line in f.readlines())
        logger.debug(f"successfully load {len(data)} raw redial data from {file_name}")

        return data

    @abstractmethod
    def _preprocess(self, train, valid, test):
        """Preprocessing the raw Redial dataset"""
        pass
