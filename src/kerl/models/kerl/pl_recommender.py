import json
from typing import List

import torch
import wandb
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torchmetrics import MeanMetric
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup

from .recommender import KERLRecommender
from ..utils import ContrastiveLoss
from ...metrics import get_rec_metrics


class KERLRecLight(LightningModule):
    def __init__(
            self,
            item_entity_ids: List[int],
            entity_description,
            edge,
            use_description: bool,
            lr: float,
            weight_decay: float,
            bart_lr: float,
            n_entity: int,
            n_relation: int,
            n_base: int,
            vocab_size: int,
            use_context: bool,
            text_pooling_type: str,
            text_pooling_head: int,
            kg_embedding_dim: int = 128,
            learn_pos_embeds: bool = True,
            max_n_positions: int = 30,
            use_kg_loss: bool = True,
            kg_margin: float = 1.0,
            warmup_steps: int = 1000,
            total_steps: int = None,
            num_cycles: float = 1,
            phase: str = "rec",
    ):
        super(KERLRecLight, self).__init__()

        self.use_description = use_description
        self.use_context = use_context
        self.use_kg_loss = use_kg_loss
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_cycles = num_cycles
        self.bart_lr = bart_lr
        if phase not in ["rec", "pretrain"]:
            raise ValueError("phase must be either rec or pretrain")
        self.phase = phase

        self.rec_model = KERLRecommender(
            edge=edge,
            n_entity=n_entity,
            n_relation=n_relation,
            n_base=n_base,
            vocab_size=vocab_size,
            use_context=self.use_context,
            text_pooling_type=text_pooling_type,
            text_pooling_head=text_pooling_head,
            kg_embedding_dim=kg_embedding_dim,
            learn_pos_embeds=learn_pos_embeds,
            use_kg_loss=use_kg_loss,
            kg_margin=kg_margin,
            use_description=self.use_description,
            entity_description=entity_description,
            max_n_positions=max_n_positions,
        )

        # metrics
        self.rec_criterion = nn.CrossEntropyLoss()
        # TODO: make this a hyperparameter, not hard-coded
        self.use_cl_loss = True
        self.contrastive_criterion = ContrastiveLoss(0.07)
        self.cl_weight = 0.5
        self.train_avg_loss = MeanMetric()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/bart-base", use_fast=True)

        self.val_avg_loss = MeanMetric()
        if self.use_kg_loss:
            self.kg_criterion = nn.LogSigmoid()
            self.kg_avg_loss = MeanMetric()
        if self.use_cl_loss:
            self.train_avg_cl_loss = MeanMetric()
            self.val_avg_cl_loss = MeanMetric()
            self.train_total_loss = MeanMetric()

        self.new_item_entity_ids = item_entity_ids.copy()
        metrics_list = get_rec_metrics(item_ids=self.new_item_entity_ids)
        self.val_metrics = nn.ModuleList([metric.clone(prefix=f"val/{self.phase}/") for metric in metrics_list])
        self.test_metrics = nn.ModuleList([metric.clone(prefix=f"test/{self.phase}/") for metric in metrics_list])
        self.target_metrics = [f"val/{self.phase}/HitRate@1", f"val/{self.phase}/HitRate@50"]

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "layernorm_embedding"]
        base_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.rec_model.named_parameters() if all(
                        nd not in n for nd in no_decay) and p.requires_grad and "bart_encoder" not in n
                ],
                "weight_decay": self.weight_decay,
                "name": "base"},
            {
                "params": [
                    p for n, p in self.rec_model.named_parameters() if any(
                        nd in n for nd in no_decay) and p.requires_grad and "bart_encoder" not in n
                ],
                "weight_decay": 0.0,
                "name": "base_no_decay"},
        ]

        optimizer = torch.optim.AdamW(
            base_grouped_parameters,
            lr=self.lr,
        )

        if self.use_context:
            bart_params = [
                {
                    "params": [
                        p for n, p in self.rec_model.named_parameters() if all(
                            nd not in n for nd in no_decay) and p.requires_grad and "bart_encoder" in n],
                    "weight_decay": self.weight_decay,
                    "lr": self.bart_lr,
                    "name": "bart_encoder_decay"},
                {
                    "params": [
                        p for n, p in self.named_parameters() if any(
                            nd in n for nd in no_decay) and p.requires_grad and "bart_encoder" in n],
                    "weight_decay": 0.0,
                    "lr": self.bart_lr,
                    "name": "bart_encoder_no_decay"},
            ]
            for param in bart_params:
                optimizer.add_param_group(param)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            # 0.1 of total train_steps
            num_warmup_steps=self.warmup_steps,
            num_cycles=self.num_cycles,
            num_training_steps=self.total_steps)

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "name": f"lr/{self.phase}",
            # "monitor": "val/rec/epoch_loss",
            "frequency": 1, }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def forward(self, batch):
        return self.rec_model(batch['context_entities'], batch['context_tokens'])

    def training_step(self, batch, batch_idx):
        if self.phase == "pretrain":
            rec_batch, kg_batch = batch["rec"], batch["kg"]
            # kg loss
            pos_heads, pos_tails = kg_batch["pos_head"], kg_batch["pos_tail"]
            rels = kg_batch["relation"]
            neg_heads, neg_tails = kg_batch["neg_head"], kg_batch["neg_tail"]
            p_score, n_score = self.rec_model.calc_kg_loss(pos_heads, rels, pos_tails, neg_heads, neg_tails)
            loss = self.calc_kg_loss(p_score, n_score)
            self.kg_avg_loss.update(loss)
            self.log(f"train/{self.phase}/kg_step_loss", loss, prog_bar=False)

            # cl_loss
            if self.use_cl_loss:
                context_entities = rec_batch['context_entities']
                context_tokens = rec_batch['context_tokens']
                kg_embeds, ct_embeds = self.rec_model.get_proj_kg_context(context_entities, context_tokens)
                cl_loss = self.contrastive_criterion(kg_embeds, ct_embeds)
                self.train_avg_cl_loss.update(cl_loss)
                self.log(f"train/{self.phase}/cl_step_loss", cl_loss, prog_bar=False)
                loss = loss + cl_loss * self.cl_weight
                self.train_total_loss.update(loss)
                self.log(f"train/{self.phase}/step_total_loss", loss, prog_bar=False)
            return loss

        else:
            context_entities = batch['context_entities']
            context_tokens = batch['context_tokens']
            labels = batch['item']
            outputs = self.rec_model(context_entities, context_tokens)
            loss = self.rec_criterion(outputs, labels)
            self.train_avg_loss.update(loss)
            self.log("train/rec/step_loss", loss, prog_bar=False)
            return loss

    def validation_step(self, batch, batch_idx):
        context_entities = batch['context_entities']
        context_tokens = batch['context_tokens']
        labels = batch['item']

        outputs = self.rec_model(context_entities, context_tokens, mode="eval")
        loss = self.rec_criterion(outputs, labels)
        self.val_avg_loss.update(loss)
        for metric in self.val_metrics:
            metric.update(outputs, labels)
        if self.phase == "pretrain" and self.use_cl_loss:
            kg_embeds, ct_embeds = self.rec_model.get_proj_kg_context(context_entities, context_tokens, mode="eval")
            cl_loss = self.contrastive_criterion(kg_embeds, ct_embeds)
            self.val_avg_cl_loss.update(cl_loss)
            self.log(f"val/{self.phase}/step_cl_loss", cl_loss, prog_bar=False)

        return loss

    def test_step(self, batch, batch_idx):
        context_entities = batch['context_entities']
        context_tokens = batch['context_tokens']
        labels = batch['item']

        outputs = self.rec_model(context_entities, context_tokens, mode="test")

        for metric in self.test_metrics:
            metric.update(outputs, labels)

    def training_epoch_end(self, outs):
        if self.phase == "pretrain":
            self.log(f"train/{self.phase}/kg_epoch_loss", self.kg_avg_loss.compute())
            self.kg_avg_loss.reset()
            if self.use_cl_loss:
                self.log(f"train/{self.phase}/cl_epoch_loss", self.train_avg_cl_loss.compute())
                self.log(f"train/{self.phase}/epoch_total_loss", self.train_total_loss.compute())
                self.train_avg_cl_loss.reset()
                self.train_total_loss.reset()
        else:
            self.log(f"train/{self.phase}/epoch_loss", self.train_avg_loss.compute())
            self.log(f"train/{self.phase}/epoch", self.current_epoch)
            self.train_avg_loss.reset()

    def on_train_start(self) -> None:
        # precompute text embeddings
        self.rec_model.text_embeds = self.rec_model.compute_text_embeds()
        self.rec_model.text_embeds.requires_grad = False

    def on_validation_start(self) -> None:
        self.rec_model.save_entity_embedding()
        if self.phase == "rec" and self._should_use_wandb_logger():
            # if use wandb logger, use wandb define metrics
            self._define_wandb_metrics()

    def _should_use_wandb_logger(self) -> bool:
        return self.trainer.current_epoch == 0 and any(isinstance(
            logger, WandbLogger) for logger in self.trainer.loggers)

    def _define_wandb_metrics(self) -> None:
        self.print("Using wandb logger, define metrics")
        wandb.define_metric(f"val/rec/epoch_loss", summary="min")
        metrics_keys = [key for metrics in self.val_metrics for key in metrics.keys()]
        for key in metrics_keys:
            wandb.define_metric(key, summary="max")

    def on_test_start(self):
        # save kg embeddings so that we can use them for evaluation and not have to recompute them
        self.rec_model.save_entity_embedding()

    def validation_epoch_end(self, outs):
        target_metrics = 0
        for metric in self.val_metrics:
            results = metric.compute()
            self.log_dict(results, prog_bar=True)
            for key in self.target_metrics:
                if key in results:
                    target_metrics += results[key]
            metric.reset()
        val_loss = self.val_avg_loss.compute()
        self.log(f"val/{self.phase}/target_metric", target_metrics)
        self.log(f"val/{self.phase}/epoch_loss", val_loss, prog_bar=True)
        self.log(f"val/{self.phase}/epoch", self.current_epoch)
        self.val_avg_loss.reset()
        if self.phase == "pretrain" and self.use_cl_loss:
            val_cl_loss = self.val_avg_cl_loss.compute()
            self.log(f"val/{self.phase}/epoch_cl_loss", val_cl_loss)
            self.val_avg_cl_loss.reset()

        return val_loss

    def test_epoch_end(self, outs):
        for metric in self.test_metrics:
            self.log_dict(metric.compute(), prog_bar=True)
            metric.reset()

    def calc_kg_loss(self, p_score, n_score):
        # - (mar - p) - (n - m))
        # - (mar - p) - (-(m-n))
        p_loss = self.kg_criterion(p_score)
        # margin - n_score = - (margin - n_score)
        n_loss = self.kg_criterion(-n_score).mean(dim=1)
        # average batch loss
        return ((-p_loss.mean()) - n_loss.mean()) / 2.0
