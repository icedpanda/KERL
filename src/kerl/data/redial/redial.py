import copy

import json
import os
import pickle
import re
import nltk


import pandas as pd
import torch
from loguru import logger
from nltk.corpus import stopwords
from tqdm import tqdm


from .base import RedialDataset
from ..utils import _restore_data
from ..utils import _save_data
from ..utils import get_mapped_entity


class WikiRedial(RedialDataset):
    def __init__(
            self, opts, tokenizer, restore=False,
            redial_path="data/redial"):
        super().__init__(opts, tokenizer, restore, redial_path)

        self.max_entity = 0
        if not self.restore:
            train_data, valid_data, test_data = self._load_dataset()
            self.train, self.valid, self.test, self.kg, self.vocab = self._preprocess(train_data, valid_data, test_data)
            data = (self.train, self.valid, self.test, self.kg, self.vocab)
            _save_data(path=self.datafolder_path, data=data)
        else:
            self.train, self.valid, self.test, self.kg, self.vocab = _restore_data(path=self.datafolder_path)

        logger.info("successfully load redial dataset with wiki KG")
        logger.error(f"max entity: {self.vocab['max_entity']}")

    def _load_kg(self):
        """
        Load KG from file
        :return:
        """
        with open(os.path.join(self.datafolder_path, "wiki/structured_kg.pickle"), "rb") as f:
            self.edges = pickle.load(f)

        with open(os.path.join(self.datafolder_path, "wiki/entity2id.json"), "r") as f:
            self.entity2id = json.load(f)

        with open(os.path.join(self.datafolder_path, "wiki/id2entity.json"), "r") as f:
            self.id2entity = json.load(f)

        with open(os.path.join(self.datafolder_path, "wiki/relations2id.json"), "r") as f:
            self.relation2id = json.load(f)

        with open(os.path.join(self.datafolder_path, "wiki/id2relation.json"), "r") as f:
            self.id2relation = json.load(f)

        with open(os.path.join(self.datafolder_path, "wiki/id2description.json"), "r") as f:
            self.id2description = json.load(f)

        with open(os.path.join(self.datafolder_path, "wiki/sent2entity.json"), "r") as f:
            self.sent2entity = json.load(f)

        movie = pd.read_csv(os.path.join(self.datafolder_path, "raw_redial_dataset/movies_with_mentions.csv"))
        movie_ids = movie["movieId"].tolist()
        # n classes
        self.n_classes = len([movie for movie in movie_ids if str(movie) in self.entity2id]) + 1

        return {
            "edges": self.edges,
            "n_classes": self.n_classes,
            "id2description": self.id2description, }

    def _load_vocab(self):
        return {
            "entity2id": self.entity2id,
            "id2entity": self.id2entity,
            "relation2id": self.relation2id,
            "id2relation": self.id2relation,
            "n_entity": len(self.entity2id) + 1,
            "n_relation": len(self.relation2id),
            "vocab_size": len(self.tokenizer),
            "pad": self.tokenizer.pad_token_id,
            "start": self.tokenizer.cls_token_id,
            "end": self.tokenizer.sep_token_id,
            "unk": self.tokenizer.unk_token_id,
            "pad_entity": 0,
            "max_entity": self.max_entity,
            "kg_vocab": self.create_description_token_mask(),
            "n_classes": self.n_classes,
        }

    def _preprocess(self, train, valid, test):
        kg = self._load_kg()
        vocab = self._load_vocab()

        preprocessed_train = self._build_dialog(train)
        preprocessed_valid = self._build_dialog(valid)
        preprocessed_test = self._build_dialog(test)
        logger.debug("successfully build redial dataset")

        train = self._convert_dialog_to_utterance(preprocessed_train)
        valid = self._convert_dialog_to_utterance(preprocessed_valid)
        test = self._convert_dialog_to_utterance(preprocessed_test)

        return train, valid, test, kg, vocab

    def _build_dialog(self, data):
        data_set = []
        logger.debug("Building raw dialog from Redial dataset")
        for instance in tqdm(data):
            dialog = []
            messages = instance["messages"]
            initiator_id = instance["initiatorWorkerId"]
            respondent_id = instance["respondentWorkerId"]
            movies2name = instance["movieMentions"]
            for utt_idx, message in enumerate(messages):
                if message["senderWorkerId"] == initiator_id:
                    role = "Seeker"
                elif message["senderWorkerId"] == respondent_id:
                    role = "Recommender"
                else:
                    raise ValueError("Invalid senderWorkerId")
                if message["text"] == "":
                    break
                raw_text = message["text"]
                # use nltk to tokenize the text
                entity = get_mapped_entity(raw_text, self.sent2entity)
                temp_utterance = {
                    "utt_id": utt_idx,
                    "role": role,
                    "raw_text": raw_text,
                    "entity": entity,
                }
                dialog.append(temp_utterance)
            conversation = {
                "conv_id": instance["conversationId"],
                "dialog": dialog,
                "movies2name": movies2name,
            }
            data_set.append(conversation)

        return data_set

    def _convert_ids_to_indices(self, text, mask=True, movie2name=None):
        """@movieID -> @movieIdx"""
        pattern = re.compile("@\\d+")
        movie_id_list = []

        def convert(match):
            # [@movieID, movieID]
            movie_id = match.group(0)
            entity = str(movie_id[1:])
            if entity is not None:
                movie_id_list.append(entity)
            if mask:
                return self.item_mask
            else:
                return movie2name[entity] if entity in movie2name else self.item_mask

        return re.sub(pattern, convert, text), movie_id_list,

    def _convert_dialog_to_utterance(self, raw_data):
        """
        Convert the dialog to utterances and merged it
        e.g. sample 1: utterance1,
        sample 2: utterance1 + utterance2, sample
        3: utterance1 + utterance2 + utterance3
        """
        augmented_dialog = [
            self._merge_dialog(
                conversation["dialog"],
                conversation["movies2name"]) for conversation in raw_data]

        return [self._augment_and_add(conv) for conv in augmented_dialog]

    def _merge_dialog(self, dialog, movies2name):
        """
        Merge the utterances into one utterance if the message is sent by
        same person.
        Convert movie/entity to id
        i.e.[seeker utterance 1, seeker utterance 2, recommender utterance 1,
        seeker utterance 3] ->
        [seeker utterance 1+2, recommender utterance 1, seeker utterance 3]
        """
        augmented_dialog = []
        last_role = None

        for utterance in dialog:
            text = utterance["raw_text"]
            entity_ids = [self.entity2id[entity] for entity in utterance["entity"]]

            if utterance["role"] == last_role:
                augmented_dialog[-1]["text"] += " " + text
                augmented_dialog[-1]["entity"] += entity_ids
            else:
                augmented_dialog.append({
                    "role": utterance["role"],
                    "text": text,
                    "entity": entity_ids,
                    "movies2name": movies2name},
                )
            last_role = utterance["role"]
        return augmented_dialog

    def _augment_and_add(self, raw_dialog):
        augmented_dialog_dicts = []
        # add start token
        # context = self.tokenizer.bos_token
        context = ""
        context_entities, context_items = [], []
        entity_set = set()
        first_utt = True
        for i, conv in enumerate(raw_dialog):
            text, entities = conv["text"], conv["entity"]
            role_prompt = "Recommender" if conv["role"] == "Recommender" else "User"
            mask_text, movies = self._convert_ids_to_indices(text)
            mask_text = f"{role_prompt}: {mask_text}"
            utt, _ = self._convert_ids_to_indices(text, mask=False, movie2name=conv["movies2name"])
            movies = [self.entity2id[movie] for movie in movies if movie in self.entity2id]
            if not first_utt:
                conv_dict = {
                    "role": conv['role'],
                    "context": copy.deepcopy(context),
                    "raw_text": copy.deepcopy(text),
                    "response": copy.deepcopy(mask_text),
                    "context_entities": copy.deepcopy(context_entities),
                    "context_items": copy.deepcopy(context_items),
                    "items": movies, }

                augmented_dialog_dicts.append(conv_dict)
            # text_tokens becomes context_tokens for next round since it becomes history
            prompt = "User: " if conv['role'] == "Seeker" else "Recommender: "
            context += prompt + utt + self.tokenizer.sep_token
            first_utt = False
            context_items += movies
            for entity in entities + movies:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)
            if len(context_entities) > self.max_entity:
                self.max_entity = len(context_entities)

        return augmented_dialog_dicts

    def create_description_token_mask(self):

        entity_description = [item for key, item in self.id2description.items()]
        all_descriptions = []
        for description in entity_description:
            tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(description))
            all_descriptions.extend(tokens)
        unique_tokens = list(set(all_descriptions))
        return self._create_mask(unique_tokens, len(self.tokenizer))

    def _create_mask(self, candidate_list, num_vocab):

        token_mask = torch.zeros(num_vocab, dtype=torch.bool)
        for i in range(num_vocab):
            if i in candidate_list and i not in self.tokenizer.all_special_ids:
                token_mask[i] = True

        return token_mask
