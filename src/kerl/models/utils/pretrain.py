import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    def __init__(
            self,
            p_norm=1,
            norm_flag=True,
            margin=1,):
        """
        :param p_norm: normalization
        :param norm_flag: norm_flag
        """
        super(TransE, self).__init__()
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.margin = margin

    def forward(self, head, relation, tail):
        if self.norm_flag:
            head = F.normalize(head, p=2, dim=-1)
            relation = F.normalize(relation, p=2, dim=-1)
            tail = F.normalize(tail, p=2, dim=-1)
        score = head + relation - tail
        return self.margin - torch.norm(score, p=self.p_norm, dim=-1)


class KGLoss(nn.Module):
    def __init__(
            self,
            p_norm=1,
            norm_flag=True,
            margin=1,
            pretrain_type="transE"):
        super(KGLoss, self).__init__()

        self.pretrain_type = pretrain_type
        self.score = TransE(p_norm=p_norm, norm_flag=norm_flag, margin=margin)

    def forward(self, head, relation, tail):
        return self.score(head, relation, tail)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function from https://github.com/google-research/simclr.
    """
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, feature1, feature2):
        # view1, view2: batch_size, hidden_size
        features = torch.cat([feature1, feature2], dim=0)
        bs = feature1.shape[0]

        # create labels
        labels = torch.cat([torch.arange(bs) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        # similarity matrix
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        assert similarity_matrix.shape == (2 * bs, 2 * bs)
        assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        return self._compute_loss(positives, negatives)

    def _compute_loss(self, positives, negatives):
        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return self.loss_func(logits, labels)
