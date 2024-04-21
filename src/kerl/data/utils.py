import os
import pickle as pkl
from copy import copy
from typing import List
from typing import Optional
from typing import Union

import torch
from loguru import logger


def _restore_data(path, file_name="all_data.pkl"):
    """
    Restore saved dataset.
    :param path: path to saved dataset
    :param file_name:  file of saved dataset. Defaults to "all_data.pkl".
    :return:
    """
    if not os.path.exists(os.path.join(path, file_name)):
        raise ValueError(f'Saved dataset [{file_name}] does not exist')
    with open(os.path.join(path, file_name), 'rb') as f:
        dataset = pkl.load(f)
    logger.info(f'Restore dataset from [{file_name}]')
    return dataset


def _save_data(data, path, file_name="all_data.pkl"):
    save_path = os.path.join(path, file_name)
    with open(save_path, 'wb') as f:
        pkl.dump(data, f)
    logger.info(f'[Save dataset to {file_name}]')


def add_start_end_token_idx(
        vec: list,
        start_token_idx: int = None,
        end_token_idx: int = None
):
    """
    Can choose to add start token in the beginning and end token in the end.

    Args:
        vec: source list composed of indexes.
        start_token_idx: index of start token.
        end_token_idx: index of end token.

    Returns:
        list: list added start or end token index.

    """
    res = copy(vec)
    res.insert(0, start_token_idx)
    res.append(end_token_idx)
    return res


def padded_tensor(
        items: List[Union[List[int], torch.LongTensor]],
        pad_idx: int = 0,
        pad_tail: bool = True,
        max_len: Optional[int] = None,
) -> torch.LongTensor:
    """
    Create a padded matrix from an uneven list of lists.

    Returns padded matrix.

    Matrix is right-padded (filled to the right) by default, but can be
    left padded if the flag is set to True.

    Matrix can also be placed on cuda automatically.

    :param list[iter[int]] items: List of items
    :param int pad_idx: the value to use for padding
    :param bool pad_tail:
    :param int max_len: if None, the max length is the maximum item length

    :returns: padded tensor.
    :rtype: Tensor[int64]

    """
    # number of items
    n = len(items)
    # length of each item
    lens: List[int] = [len(item) for item in items]  # type: ignore
    # max in time dimension
    t = max(lens) if max_len is None else max_len
    # if input tensors are empty, we should expand to nulls
    t = max(t, 1)

    if isinstance(items[0], torch.Tensor):
        # keep type of input tensors, they may already be cuda ones
        output = items[0].new(n, t)  # type: ignore
    else:
        output = torch.LongTensor(n, t)  # type: ignore
    output.fill_(pad_idx)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            # skip empty items
            continue
        if not isinstance(item, torch.Tensor):
            # put non-tensors into a tensor
            item = torch.tensor(item, dtype=torch.long)  # type: ignore
        if pad_tail:
            # place at beginning
            output[i, :length] = item
        else:
            # place at end
            output[i, t - length:] = item

    return output


def conv_truncate(tokens, max_len, end_token_idx, start_token_idx):
    """
    Truncate tokens to make its length no more than max_len

    :param tokens: list of tokens
    :param max_len: max length of tokens
    :param end_token_idx: index of end token
    :param start_token_idx: index of start token
    :return: truncated tokens
    """
    num_tokens = len(tokens)
    if num_tokens <= max_len:
        return tokens
    dialog_token = copy(tokens)

    # find last complete utterance
    # save space for end token and start token insert later on
    last_complete_dialog = dialog_token[-(max_len - 2):]
    # last_end_token_idx = last_complete_dialog.index(end_token_idx)
    # last_complete_dialog = last_complete_dialog[last_end_token_idx + 1:]
    dialog = add_start_end_token_idx(last_complete_dialog, start_token_idx, end_token_idx)

    assert len(dialog) <= max_len, "Truncated dialog is longer than max_len"
    return dialog


def get_mapped_entity(sentence, sent2entity):
    """
    get entity from sentence

    :param sentence: sentence
    :param sent2entity: sentence to entity mapping
    :return: entity: list of entity

    """
    entity = []
    # sentence = sentence.lower()
    return sent2entity[sentence] if sentence in sent2entity else entity
