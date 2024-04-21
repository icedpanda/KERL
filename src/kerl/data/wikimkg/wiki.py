import random
from typing import Dict
from typing import List
import itertools
import torch
from transformers import AutoTokenizer
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .. import WikiRedial
from ..utils import conv_truncate
from ..utils import padded_tensor


class WikiMKGDataset(Dataset):

    def __init__(
            self, opt, dataset: List = None, vocab: Dict = None,
            edge: List = None, stage: str = "rec", tokenizer=None):

        self.opt = opt
        self.stage = stage
        if self.stage == "kg":
            self.edges = edge
        else:
            self.tokenizer = tokenizer
            self.pad_entity_idx = vocab["pad_entity"]
            self.start_token_idx = vocab["start"]
            self.end_token_idx = vocab["end"]
            self.context_max_length = opt["context_truncate"]
            self.response_max_length = opt["response_truncate"]
            self.max_entity_length = opt["max_entity_length"]
            if self.context_max_length > self.tokenizer.model_max_length:
                raise ValueError("context_max_len cannot be larger than model_max_length")
            if self.context_max_length < 200:
                raise ValueError("context_max_len should be larger than 200 ")
            if self.stage == "rec":
                self.rec_data = self._augment_data(dataset, self.stage)
            elif self.stage == "conv":
                self.conv_data = self._augment_data(dataset, self.stage)

    def __len__(self):
        if self.stage == "rec":
            return len(self.rec_data)
        if self.stage == "conv":
            return len(self.conv_data)
        if self.stage == "kg":
            return len(self.edges)

    def __getitem__(self, idx):
        if self.stage == "rec":
            return self.rec_data[idx]
        if self.stage == "conv":
            return self.conv_data[idx]
        if self.stage == "kg":
            return self.edges[idx]

    def _augment_data(self, dataset, stage):
        if stage == "rec":
            return list(itertools.chain(*[self._augment_rec_data(utterance)
                                          for conv_dict in dataset
                                          for utterance in conv_dict
                                          if utterance['role'] == 'Recommender'
                                          and utterance['items']]))
        else:
            return [self._augment_conv_data(utterance)
                    for conv_dict in dataset
                    for utterance in conv_dict
                    if utterance['role'] == 'Recommender']

    def _augment_rec_data(self, utterance):
        """
        Augment the dataset with movies that are recommended by the
        recommender.
        The last utterance of the conversation should be recommender since model
        need to make the prediction.
        """

        augment_list = []
        context_token_ids = self.tokenizer.encode(utterance['context'], add_special_tokens=True)

        context_token_ids = conv_truncate(context_token_ids,
                                          self.context_max_length,
                                          self.end_token_idx,
                                          self.start_token_idx)
        for movie in utterance['items']:
            augment_conv_dict = {
                'context_tokens': context_token_ids,
                'context_entities': utterance['context_entities'][-self.max_entity_length:],
                'item': movie, }
            augment_list.append(augment_conv_dict)
        return augment_list

    def _augment_conv_data(self, utterance):
        """
        truncate the context and response here rather than doing it in
        dataloader batch_fn so that we don't
        have to do it in every epoch.
        :param utterance: one conversation dict
        :return: truncated conversation dict
        """

        context_token_ids = self.tokenizer.encode(utterance['context'], add_special_tokens=False)
        # truncate context and add special tokens here
        context_token_ids = conv_truncate(context_token_ids,
                                          self.context_max_length,
                                          self.end_token_idx,
                                          self.start_token_idx)

        resp_token_ids = self.tokenizer.encode(
            utterance['response'],
            max_length=self.response_max_length,
            add_special_tokens=True,)

        return {
            "context_entities": utterance['context_entities'][-self.max_entity_length:],
            "context_tokens": context_token_ids,
            "response": resp_token_ids,
            "raw_response": utterance['raw_text']
        }


class WikiMKGDataModule(LightningDataModule):
    def __init__(
            self, opt, dataset: WikiRedial, batch_size, tokenizer,
            stage, kg_batch_size=64, neg_batch_size=8, use_kg=False):
        super().__init__()
        # random.seed(42)

        self.batch_size = batch_size
        self.use_kg = use_kg

        self.tokenizer = tokenizer
        self.stage = stage
        self.pad_entity_idx = dataset.vocab["pad_entity"]
        self.max_response_length = opt.get("response_truncate", 30)
        if self.stage == "conv":
            self.kg_vocab_mask = dataset.vocab["kg_vocab"]

        self.train_dataset = WikiMKGDataset(
            opt=opt,
            dataset=dataset.train,
            vocab=dataset.vocab,
            stage=self.stage,
            tokenizer=self.tokenizer,
        )

        self.val_dataset = WikiMKGDataset(
            opt=opt,
            dataset=dataset.valid,
            vocab=dataset.vocab,
            stage=self.stage,
            tokenizer=self.tokenizer,
        )

        self.test_dataset = WikiMKGDataset(
            opt=opt,
            dataset=dataset.test,
            vocab=dataset.vocab,
            stage=self.stage,
            tokenizer=self.tokenizer,)
        if self.use_kg:
            # negative samples per positive sample
            # 1 positive sample
            # neg_batch_size * negative tail + neg_batch_size * negative head
            self.kg_batch_size = kg_batch_size
            self.neg_batch_size = neg_batch_size // 2
            self.edge_dataset = WikiMKGDataset(opt, edge=dataset.kg["edges"], stage="kg")
            self.all_edge_set = set(self.edge_dataset.edges)
        self.descr_truncate = opt.get("descr_truncate", 30)
        self.entity_description = self._prepare_description(dataset.kg["id2description"])

    def train_dataloader(self):
        if self.stage == "conv":
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True,
                shuffle=True,
                collate_fn=self.conv_collate_fn)
        elif self.stage == "pretrain":
            return self.get_pretrain_loader()
        else:
            return self.get_rec_loader()

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 4,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.conv_collate_fn if self.stage == "conv" else
            self.rec_collate_fn)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 4,
            num_workers=2,
            shuffle=False,
            collate_fn=self.conv_collate_fn if self.stage == "conv" else
            self.rec_collate_fn)

    def get_kg_loader(self):
        return DataLoader(
            self.edge_dataset,
            batch_size=self.kg_batch_size,
            num_workers=4,
            persistent_workers=False,
            pin_memory=False,
            shuffle=True,
            collate_fn=self.kg_collate_fn)

    def get_rec_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            shuffle=True,
            collate_fn=self.rec_collate_fn)

    def get_pretrain_loader(self):
        return {"rec": self.get_rec_loader(),
                "kg": self.get_kg_loader()
                }

    def change_stage(self, stage):
        self.stage = stage

    def get_total_steps(self, dataset):
        if dataset in ["rec", "conv"]:
            return len(self.train_dataset) // self.batch_size
        if dataset == "kg":
            return len(self.edge_dataset) // self.kg_batch_size
        else:
            raise ValueError("dataset should be rec or kg")

    def rec_collate_fn(self, batch):
        context_entities, item = [], []
        context_batch = {"input_ids": []}
        for rec_dict in batch:
            context_entities.append(rec_dict['context_entities'])
            item.append(rec_dict['item'])
            context_batch["input_ids"].append(rec_dict['context_tokens'])

        context_entities = padded_tensor(context_entities, self.pad_entity_idx, pad_tail=True)
        item = torch.tensor(item, dtype=torch.long)
        context_batch = self.tokenizer.pad(context_batch, return_tensors="pt", padding=True, pad_to_multiple_of=8, )

        return {
            "context_entities": context_entities,
            "context_tokens": context_batch,
            "item": item,
        }

    def conv_collate_fn(self, batch):
        response_batch, context_batch = {"input_ids": []}, {"input_ids": []}
        context_entities, raw_text_list = [], []
        for conv_dict in batch:
            context_batch["input_ids"].append(conv_dict['context_tokens'])
            response_batch["input_ids"].append(conv_dict['response'])
            context_entities.append(conv_dict['context_entities'],)
            raw_text_list.append(conv_dict['raw_response'])

        batch_response = self.tokenizer.pad(response_batch, padding=True, return_tensors="pt", )

        batch_response["input_ids"] = torch.tensor(batch_response["input_ids"], dtype=torch.long)
        return {
            "context_tokens": self.tokenizer.pad(context_batch,
                                                 padding=True,
                                                 # fp16
                                                 pad_to_multiple_of=8,
                                                 return_tensors="pt", ),
            "context_entities": padded_tensor(
                context_entities,
                self.pad_entity_idx,
                pad_tail=True),
            "response": batch_response,
            "raw_text": raw_text_list, }

    def kg_collate_fn(self, batch):
        # generate negative samples
        neg_heads, neg_tails = self.generate_neg_samples(batch)
        head, tail, relation = [], [], []
        for triple in batch:
            head.append(triple[0])
            tail.append(triple[1])
            relation.append(triple[2])

        return {
            "pos_head": torch.tensor(head, dtype=torch.long),
            "pos_tail": torch.tensor(tail, dtype=torch.long),
            "relation": torch.tensor(relation, dtype=torch.long),
            "neg_head": torch.tensor(neg_heads, dtype=torch.long),
            "neg_tail": torch.tensor(neg_tails, dtype=torch.long), }

    def generate_neg_samples(self, current_batch):
        batch_neg_heads, batch_neg_tails = [], []

        for edge in current_batch:
            neg_tails = []
            neg_heads = []
            # random sample edges
            for _ in range(self.neg_batch_size):
                neg_head = self._get_random_neg(edge, mode="head")
                neg_heads.append(neg_head)

                neg_tail = self._get_random_neg(edge, mode="tail")
                neg_tails.append(neg_tail)

            batch_neg_heads.append(neg_heads)
            batch_neg_tails.append(neg_tails)
        neg_sample_len = len(batch_neg_heads[0]) + len(batch_neg_tails[0])
        assert neg_sample_len == self.neg_batch_size * 2
        return batch_neg_heads, batch_neg_tails

    def _get_random_neg(self, pos_edge, mode):

        pos_head = pos_edge[0]
        pos_tail = pos_edge[1]
        relation = pos_edge[2]
        neg_item = None
        while neg_item is None:
            random_edge = random.choice(self.edge_dataset.edges)
            if mode == "head":
                neg_item = random_edge[0]
                # make sure the new neg_pairs are not in the original graph
                if (neg_item, pos_tail, relation) in self.all_edge_set:
                    neg_item = None
            else:
                neg_item = random_edge[1]
                if (pos_head, neg_item, relation) in self.all_edge_set:
                    neg_item = None

        return neg_item

    def _prepare_description(self, description):
        """
        Prepare description for each entity
        :param description:
        :return:
        """
        id2description_list = [value for key, value in description.items()]
        # insert empty string for padding since index 0 is entity padding idx
        id2description_list.insert(0, "")
        # TODO: don't hard code the tokenizer
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini", use_fast=True)
        return tokenizer(
            id2description_list,
            max_length=self.descr_truncate,
            padding=True,
            truncation=True,
            return_tensors="pt")
