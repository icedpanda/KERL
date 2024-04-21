import copy
import csv

import json
import os
import re

from typing import Tuple
from typing import List


import pandas as pd
import torch
from loguru import logger

from tqdm import tqdm


from .base import InspiredDataset
from ..utils import _restore_data
from ..utils import _save_data

from ..utils import get_mapped_entity


class WikiInspired(InspiredDataset):
    def __init__(self, opts, tokenizer, restore=False, datafolder_path="data/inspired"):
        super().__init__(opts, tokenizer, restore, datafolder_path)

        self.max_entity = 0
        if not self.restore:
            train_data, valid_data, test_data = self._load_dataset()
            self.train, self.valid, self.test, self.kg, self.vocab = self._preprocess(train_data, valid_data, test_data)
            data = (self.train, self.valid, self.test, self.kg, self.vocab)
            logger.warning(f"max entity: {self.max_entity}")
            _save_data(path=self.datafolder_path, data=data)
        else:
            self.train, self.valid, self.test, self.kg, self.vocab = _restore_data(path=self.datafolder_path)
            logger.warning(f"max entity: {self.vocab['max_entity']}")

        logger.info("successfully load inspired dataset with wiki KG")

    def _load_kg(self):
        """Load the KG data"""

        self.edges = self._load_structured_kg()

        with open(os.path.join(self.datafolder_path, "wiki/entity2id.json"), "r") as f:
            self.entity2id = json.load(f)

        with open(os.path.join(self.datafolder_path, "wiki/relations2id.json"), "r") as f:
            self.relation2id = json.load(f)

        with open(os.path.join(self.datafolder_path, "wiki/id2description.json"), "r") as f:
            self.id2description = json.load(f)

        with open(os.path.join(self.datafolder_path, "wiki/sentence2entity.json"), "r") as f:
            self.sent2entity = json.load(f)

        with open(os.path.join(self.datafolder_path, "wiki/movie_kg.json"), "r") as f:
            movie_kg = json.load(f)
            movie_list = list(movie_kg.keys())

        self.n_classes = len([movie for movie in movie_list if movie in self.entity2id]) + 1
        logger.error(f"Number of classes: {self.n_classes}")

        return {
            "edges": self.edges,
            "id2description": self.id2description,
            "n_classes": self.n_classes,
        }

    def _load_structured_kg(self) -> List[Tuple[int, int, int]]:
        """
        Load the structured knowledge graph from a CSV file.

        Returns:
            edge_loaded (List[Tuple[int, int, int]]): A list of tuples. Each tuple contains three ints. The
                source, destination, and relation of an edge.
        """

        edge_loaded = set()  # Set to store the loaded edges
        file_path = os.path.join(self.datafolder_path, "wiki/edge_set.csv")  # Get the file path
        with open(file_path, "r") as f:  # Open the file for reading
            reader = csv.reader(f)  # Create a CSV reader
            next(reader)  # Skip the header row
            for row in reader:  # Iterate over each row in the file
                source, destination, relation = map(int, row)  # Convert strings to integers
                edge_loaded.add((source, destination, relation))  # Add the edge to the set

        logger.info("Successfully loaded the structured knowledge graph")
        logger.info(f"Number of edges: {len(edge_loaded)}")
        return list(edge_loaded)  # Return the list of loaded edges

    def _load_vocab(self):
        return {
            "entity2id": self.entity2id,
            "relation2id": self.relation2id,
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
        logger.debug("successfully build inspired dataset")

        train = self._convert_dialog_to_utterance(preprocessed_train, True)
        valid = self._convert_dialog_to_utterance(preprocessed_valid)
        test = self._convert_dialog_to_utterance(preprocessed_test)

        with open("preprocessed_train.json", "w") as f:
            json.dump(train, f)

        with open("preprocessed_valid.json", "w") as f:
            json.dump(valid, f)

        with open("preprocessed_test.json", "w") as f:
            json.dump(test, f)

        logger.error("tokenizer mask token: {}".format(self.tokenizer.mask_token))

        return train, valid, test, kg, vocab

    def _build_dialog(self, data):
        """Build dialog from the raw data"""
        data_set = []
        current_dialog = []
        previous_dialog_id = None
        for row in tqdm(data.itertuples(), total=len(data)):
            dialog_id = row.dialog_id
            if dialog_id != previous_dialog_id:
                # Start a new dialog if the dialog_id has changed
                if current_dialog:
                    data_set.append(current_dialog)
                current_dialog = []
            movies = row.movies
            if pd.isnull(movies):
                movies = []
            else:
                movies = movies.split(";")
                movies = [name.strip() for name in movies]

            # match with redial, so we can use the same dataset class
            role = "Recommender" if row.speaker == "RECOMMENDER" else "User"
            # create the utterance dict
            temp_utterance = {
                "utt_id": row.utt_id,
                "dialog_id": dialog_id,
                "role": role,
                "text": row.text,
                "masked_text": self.replace_movie_titles(
                    row.text,
                    movies) if role == "Recommender" and movies else row.text,
                "movie": movies,
                "utt_id ": row.utt_id,
                "entity": get_mapped_entity(
                    row.text,
                    self.sent2entity)}

            current_dialog.append(temp_utterance)
            previous_dialog_id = dialog_id

        if current_dialog:
            data_set.append(current_dialog)

        return data_set

    def _convert_dialog_to_utterance(self, raw_data, is_train=False):
        """
        Convert the dialog to utterances and merged it
        e.g. sample 1: utterance1,
        sample 2: utterance1 + utterance2, sample
        3: utterance1 + utterance2 + utterance3
        """
        augmented_dialog = [self._merge_dialog(conversation,) for conversation in raw_data]

        return [self._augment_and_add(conv, is_train) for conv in augmented_dialog]

    def _merge_dialog(self, dialog):
        """
        Merge the utterances into one utterance if the message is sent by
         the same person.
        Convert movie/entity to id
        i.e.[seeker utterance 1, seeker utterance 2, recommender utterance 1,
        seeker utterance 3] ->
        [seeker utterance 1+2, recommender utterance 1, seeker utterance 3]
        """
        augmented_dialog = []
        last_role = None

        for utterance in dialog:
            text = utterance["text"]

            entity_ids = [self.entity2id[entity] for entity in utterance["entity"]]
            movie_ids = [self.entity2id[movie] for movie in utterance["movie"] if movie]

            if utterance["role"] == last_role:
                augmented_dialog[-1]["text"] += " " + text
                augmented_dialog[-1]["entity"] += entity_ids
                augmented_dialog[-1]["movie"] += movie_ids
                augmented_dialog[-1]["masked_text"] += " " + utterance["masked_text"]
            else:
                augmented_dialog.append({
                    "text": text,
                    "masked_text": utterance["masked_text"],
                    "entity": entity_ids,
                    "role": utterance["role"],
                    "movie": movie_ids,
                })
            last_role = utterance["role"]

        return augmented_dialog

    def _augment_and_add(self, raw_dialog, is_train=False):
        """
        Add the utterance to the dialog
        """
        augmented_dialog_dicts = []
        # add start token
        # context = self.tokenizer.bos_token
        context = ""
        context_entities, context_items = [], []
        entity_set = set()
        first_utt = True
        for i, conv in enumerate(raw_dialog):
            text, entities, movies = conv["text"], conv["entity"], conv["movie"]
            role_prompt = "Recommender: " if conv["role"] == "Recommender" else "User: "
            mask_text = conv["masked_text"]
            mask_text = f"{role_prompt}: {mask_text}"
            if not first_utt:
                conv_dict = {
                    "role": conv["role"],
                    "context": copy.deepcopy(context),
                    "raw_text": copy.deepcopy(text),
                    "response": copy.deepcopy(mask_text),
                    "context_entities": copy.deepcopy(context_entities),
                    "items": copy.deepcopy(movies + entities) if is_train else copy.deepcopy(movies)  # follow unicrs
                }

                augmented_dialog_dicts.append(conv_dict)

            context_str = f"{text} {self.tokenizer.sep_token}"
            context += f"{role_prompt}{context_str}" if first_utt else f" {role_prompt}{context_str}"
            first_utt = False
            for entity in entities + movies:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)
            if len(context_entities) > self.max_entity:
                self.max_entity = len(context_entities)

        return augmented_dialog_dicts

    def replace_movie_titles(self, original_text, movie_names):
        mask_text = original_text

        for movie_name in movie_names:
            if movie_name.endswith(")") and "(" in movie_name:
                last_open_parenthesis_index = movie_name.rfind("(")
                movie_name_no_brackets = movie_name[:last_open_parenthesis_index].rstrip()
            else:
                movie_name_no_brackets = movie_name

            # If the movie name is found, replace it with "tokenizer.mask"
            pattern = re.compile(re.escape(movie_name_no_brackets), re.IGNORECASE)
            # Replace all occurrences of the movie name in the text with the placeholder
            mask_text = re.sub(pattern, self.item_mask, mask_text)

        return mask_text

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
