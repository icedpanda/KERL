
from typing import Optional

import torch
from torchmetrics import Metric
from torchmetrics import MetricCollection


class BaseRecMetric(Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    def __init__(self, labels_ids, k=1, dist_sync_on_step=False, compute_on_cpu=False):
        """

        Args:
            labels_ids: list of label ids to be used for the metric
            k: the k in "topk"
            dist_sync_on_step: This argument is bool that indicates if the metric should synchronize between different
                devices every time.
            compute_on_cpu: This argument is bool that indicates if the metric should be computed on CPU
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_cpu=compute_on_cpu)
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        # speed up by 750x times compared to list.index()
        self.k = k
        self.labels_ids = labels_ids

    def update(self, preds, labels):
        assert preds.shape[0] == labels.shape[0], "preds and labels must have the same number of samples"
        preds = preds[:, self.labels_ids]
        _, preds_rank = torch.topk(preds, self.k, dim=1)
        self._compute_metric(preds_rank, labels)

    def _compute_metric(self, preds_rank, labels):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class HitRate(BaseRecMetric):
    """
    Compute recall@k for a given set of labels and predictions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def _compute_metric(self, preds_rank, labels):
        self.total += labels.size(0)
        labels_idx = labels.unsqueeze(-1)

        # Find the matches between predicted ranks and labels.
        # If there is at least one match, then the prediction is correct.
        matches = (preds_rank == labels_idx).sum(dim=-1)
        self.correct += matches.sum()

    def compute(self):
        return self.correct.float() / self.total


class NDCG(BaseRecMetric):
    """
    Compute NDCG@k for a given set of labels and predictions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("score", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def _compute_metric(self, preds_rank, labels):
        self.total += labels.size(0)
        labels_idx = labels.unsqueeze(-1)

        # Find the matches between predicted ranks and labels
        matches = (preds_rank == labels_idx).nonzero(as_tuple=True)

        # Get the indices of the matches
        label_ranks = matches[1]
        scores = 1 / (torch.log2(label_ranks.float() + 2))
        self.score += scores.sum()

    def compute(self):
        return self.score.float() / self.total


class MMR(BaseRecMetric):
    """
    Compute MEAN RECIPROCAL RANK @k for a given set of labels and predictions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("score", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def _compute_metric(self, preds_rank, labels):
        self.total += labels.size(0)


        labels_idx = labels.unsqueeze(-1)
        matches = (preds_rank == labels_idx).nonzero(as_tuple=True)
        label_ranks = matches[1]  # get the column indices
        scores = 1 / (label_ranks.float() + 1)
        self.score += scores.sum()

    def compute(self):
        return self.score.float() / self.total


def get_rec_metrics(item_ids, is_dp_mode: bool = False):
    # TODO: Make the k values configurable
    ks = [1, 10, 50]
    metrics = []
    for k in ks:
        metric_collection = MetricCollection(
            [
                HitRate(
                    item_ids,
                    k=k,
                    dist_sync_on_step=is_dp_mode,),
                NDCG(
                    item_ids,
                    k=k,
                    dist_sync_on_step=is_dp_mode,)
            ], postfix=f"@{k}")
        metrics.append(metric_collection)

    return metrics
