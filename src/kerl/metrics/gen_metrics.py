import re
from typing import Optional, Dict, Any, Tuple, List

from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu
import torch
from torch import Tensor, tensor
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics import MetricCollection


class DistinctN(Metric):
    """
    Distinct-n are defined as the number of distinct n-grams in the reference.
    divided by the total number of words in the reference.
    """

    # Set to True if the metric during 'update' requires access to the global
    # metric
    # state for its calculations. If not, setting this to False indicates
    # that all
    # batch states are independent, and we will optimize the runtime of
    # 'forward'
    full_state_update: bool = True
    # Set to True if the metric reaches its optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    def __init__(self, n_gram, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "n_sentences",
            default=torch.tensor(0),
            dist_reduce_fx="sum")
        self.add_state("token_set", default=[])
        # self.token_set = set()
        self.n_gram = n_gram

    def update(
            self,
            preds: str,
            _: None,
    ):
        for pred in preds:
            self.n_sentences += 1
            pred_token = pred.split()
            for token in ngrams(pred_token, self.n_gram):
                self.token_set.append(token)

    def compute(self):
        # .reset() will clean the token_set
        return len(set(self.token_set)) / self.n_sentences


class Perplexity(Metric):
    r"""Perplexity measures how well a language model predicts a text sample.

    It's calculated as the average number of bits per word a model needs to represent the sample.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    total_log_probs: Tensor
    count: Tensor

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)
        if ignore_index is not None and not isinstance(ignore_index, int):
            raise ValueError(f"Argument `ignore_index` expected to either be `None` or an `int` but got {ignore_index}")
        self.ignore_index = ignore_index
        self.add_state("total_log_probs", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        total_log_probs, count = self._perplexity_update(preds, target, self.ignore_index)
        self.total_log_probs += total_log_probs
        self.count += count

    def compute(self) -> Tensor:
        """Compute the Perplexity."""
        return self._perplexity_compute(self.total_log_probs, self.count)

    def _perplexity_update(self, preds: Tensor, target: Tensor,
                           ignore_index: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        probs = F.softmax(preds.reshape(-1, preds.shape[-1]), dim=1)
        target = target.reshape(-1)

        if ignore_index is not None:
            mask = target.ne(ignore_index)
            target = target.where(target != ignore_index, torch.tensor(0, device=target.device))
        else:
            mask = torch.ones_like(target, dtype=torch.bool)

        probs = probs[:, target].diagonal()[mask]
        total_log_probs = -probs.log().sum()
        count = mask.sum()

        return total_log_probs, count

    def _perplexity_compute(self, total: Tensor, count: Tensor) -> Tensor:
        """Compute the Perplexity.

        Args:
            total: Log probabilities, summed over all samples
            count: Number of samples
        Returns:
            Perplexity
        """
        return torch.exp(total / count)


class ItemRatio(Metric):
    higher_is_better = True

    def __init__(self, item_masks: str, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.item_masks = item_masks
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: List[str], _: None) -> None:
        for pred in preds:
            items = re.findall(self.item_masks, pred)
            self.count += len(items)

        self.total += len(preds)

    def compute(self):
        return self.count / self.total

def get_gen_metrics(is_dp_mode: bool = False):
    # TODO: Make the k values configurable
    ks = [2, 3, 4]
    metrics = []
    for k in ks:
        metrics_collection = MetricCollection(
            [
                DistinctN(n_gram=k, dist_sync_on_step=is_dp_mode),
            ], postfix=f"@{k}"
        )
        metrics.append(metrics_collection)

    # add perplexity
    return metrics


def get_perplexity_metric(ignore_index: Optional[int] = None):
    return Perplexity(ignore_index=ignore_index)

