import os
from abc import ABC
from abc import abstractmethod

from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

class BaseDataset(ABC):
    AVAILABLE_TOKENIZER = {"bert-mini": "prajjwal1/bert-mini", "bart": "facebook/bart-base", }

    def __init__(self, opts, tokenizer, restore=False, datafolder_path="data/redial"):
        self.opts = opts
        self.tokenizer_name = self.AVAILABLE_TOKENIZER[tokenizer]
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        self.item_mask = self.tokenizer.mask_token
        self.datafolder_path = datafolder_path

        self.restore = restore

    @abstractmethod
    def _load_dataset(self):
        """Load the raw Redial dataset"""
        pass

    @abstractmethod
    def _load_raw_data(self, file_name):
        """Load the raw dataset """
        pass
