import copy
import json

from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.optim import AdamW
from torchmetrics import MeanMetric
from transformers import PreTrainedTokenizer
from transformers import get_cosine_schedule_with_warmup

from ...metrics import get_gen_metrics, get_perplexity_metric, ItemRatio
from .generator import KERLGenerator
from .recommender import KERLRecommender


class KERLGenLight(LightningModule):
    def __init__(
            self,
            rec_model: KERLRecommender,
            tokenizer: PreTrainedTokenizer,
            kg_vocab_mask,
            lr: float,
            weight_decay: float,
            warmup_steps: int = 1000,
            total_steps: int = None,
            num_cycles: float = 1,
            mx_resp_len: int = 30,
            output_dir: str = None,
    ):
        super(KERLGenLight, self).__init__()

        self.output_dir = output_dir
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.num_cycles = num_cycles

        self.generator = KERLGenerator(
            rec_model=rec_model,
            kg_vocab_mask=kg_vocab_mask,
            max_response_len=mx_resp_len,
        )
        self._freeze_rec()
        self.tokenizer = tokenizer
        self.gen_criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.train_avg_loss = MeanMetric()
        self.val_avg_loss = MeanMetric()
        metrics = get_gen_metrics()
        # self.test_ppl = get_perplexity_metric(ignore_index=self.tokenizer.pad_token_id)
        self.item_ratio = ItemRatio(item_masks=self.tokenizer.mask_token)
        self.val_metrics = [metric.clone(prefix="val/conv/") for metric in metrics]
        self.test_metrics = [metric.clone(prefix="test/conv/") for metric in metrics]

    def _freeze_rec(self):
        for param in self.generator.rec_model.parameters():
            param.requires_grad = False

    def configure_optimizers(self):

        no_decay = ["bias", "LayerNorm.weight", "layernorm_embedding"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.generator.named_parameters()
                    if all(nd not in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0,
                "name": "conv_model_no_decay",
            },
            {
                "params": [
                    p
                    for n, p in self.generator.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
                "name": "conv_model_decay",
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_cycles=self.num_cycles,
            num_training_steps=self.total_steps,
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "name": "lr/gen/",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def training_step(self, batch, batch_idx):
        context_entities = batch['context_entities']
        context_tokens = batch['context_tokens']
        labels = batch["response"].input_ids

        logits, _ = self.generator(
            context_entities=context_entities,
            context_tokens=context_tokens,
            labels=labels,
            mode="train",
        )
        labels = labels.view(-1)
        logits = logits.view(-1, logits.shape[-1])
        loss = self.gen_criterion(logits, labels)
        self.train_avg_loss.update(loss)
        self.log("train/conv/step_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        context_entities = batch['context_entities']
        context_tokens = batch['context_tokens']
        response = batch["response"].input_ids

        logits, preds, = self.generator(
            context_entities=context_entities,
            context_tokens=context_tokens,
            labels=response,
            mode="val",
        )
        labels = response.view(-1)
        logits = logits.view(-1, logits.shape[-1])
        loss = self.gen_criterion(logits, labels)

        self.val_avg_loss.update(loss)
        self.log("val/conv/step_loss", loss, prog_bar=False)

        pred_list, label_list = self._convert_to_text(preds), self._convert_to_text(response)
        for metric in self.val_metrics:
            metric.update(pred_list, label_list)

        return loss

    def test_step(self, batch, batch_idx):
        context_entities = batch['context_entities']
        context_tokens = batch['context_tokens']
        response = batch["response"].input_ids
        raw = batch["raw_text"]

        _, preds, = self.generator(
            context_entities=context_entities,
            context_tokens=context_tokens,
            mode="test",
        )

        pred_list, label_list = self._convert_to_text(preds), self._convert_to_text(response)
        self.item_ratio.update(pred_list, None)
        for metric in self.test_metrics:
            metric.update(pred_list, label_list)

    def on_training_epoch_end(self):
        self.log("train/conv/epoch_loss", self.train_avg_loss.compute(), prog_bar=True)
        self.log("train/conv/epoch", self.current_epoch)
        self.train_avg_loss.reset()

    def validation_epoch_end(self, outputs):
        for metric in self.val_metrics:
            self.log_dict(metric.compute(), prog_bar=True)
            metric.reset()
        val_loss = self.val_avg_loss.compute()
        self.log("val/conv/epoch_loss", val_loss, prog_bar=True)
        self.log("val/conv/epoch", self.current_epoch)
        self.val_avg_loss.reset()
        return val_loss

    def test_epoch_end(self, outputs):
        for metric in self.test_metrics:
            self.log_dict(metric.compute(), prog_bar=True)
            metric.reset()

        self.log("test/conv/item_ratio", self.item_ratio.compute(), prog_bar=True)

    def _convert_to_text(self, ids):
        text_list = self.tokenizer.batch_decode(ids, skip_special_tokens=False)
        return [
            text.replace("<s>", "")
            .replace("</s>", "")
            .replace("<pad>", "")
            .replace("<mask>", " <mask>")
            for text in text_list
        ]
