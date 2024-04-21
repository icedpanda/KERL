import os
import shutil

from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

from ..data import WikiRedial, WikiMKGDataModule
from ..models.kerl import KERLRecLight, KERLGenLight
from .base import BaseSystem


class KERLSystem(BaseSystem):
    """This is the system for KERL"""

    def __init__(self, opt, data_module: WikiRedial):
        super(KERLSystem, self).__init__(opt)

        self.setup_kerl_configs(data_module)
        self.rec_dataset = WikiMKGDataModule(
            opt=opt,
            dataset=data_module,
            tokenizer=data_module.tokenizer,
            batch_size=self.rec_batch_size,
            kg_batch_size=self.kg_batch_size,
            neg_batch_size=self.neg_batch_size,
            use_kg=self.use_kg_loss,
            stage="rec")
        self.conv_dataset = WikiMKGDataModule(
            opt=opt,
            dataset=data_module,
            tokenizer=data_module.tokenizer,
            batch_size=self.conv_batch_size,
            stage="conv")

    def setup_kerl_configs(self, data_module):

        self.vocab_size = data_module.vocab["vocab_size"]
        self.item_entity_ids = list(range(data_module.kg["n_classes"]))
        self.use_context = self.opt["use_context"]
        self.bart_lr = self.opt["rec"]["bart_lr"]
        self.text_pooling_type = self.opt["text_pooling_type"]
        self.text_pooling_head = self.opt["text_pooling_head"]

        self.setup_kg_config(data_module, self.opt["kg_config"])

    def setup_kg_config(self, data_module, kg_config):

        self.edges = data_module.kg["edges"]
        self.n_entity = data_module.vocab["n_entity"]
        self.n_relation = data_module.vocab["n_relation"]
        self.n_base = self.opt["n_bases"]
        self.kg_embedding_dim = self.opt["kg_embedding_dim"]
        self.learn_pos_embed = self.opt["position_encoding"]
        self.max_n_positions = self.opt["max_n_positions"]
        self.use_description = self.opt["use_description"]
        self.use_kg_loss = self.opt.get("use_kg_loss", True)
        self.neg_batch_size = kg_config.get("neg_batch_size", 128)
        self.kg_batch_size = kg_config.get("kg_batch_size", 512)
        self.pretrain_lr = kg_config.get("pretrain_lr", None)
        self.kg_margin = kg_config.get("kg_margin", None)
        self.pretrain_epoch = kg_config.get("epoch", None)
        self.pretrain_bart_lr = kg_config.get("pretrain_bart_lr", None)
        self.pretrain_reload = kg_config.get("pretrain_reload", None)
        self.pretrain_weight_decay = kg_config.get("weight_decay", None)
        if self.pretrain_reload:
            self.pretrain_reload_path = kg_config["pretrain_model_path"]

    def _init_lightning_recommender(self, phase="rec"):
        lr = self.rec_lr if phase == "rec" else self.pretrain_lr
        bart_lr = self.bart_lr if phase == "rec" else self.pretrain_bart_lr
        weight_decay = self.rec_weight_decay if phase == "rec" else self.pretrain_weight_decay
        self.rec_model = KERLRecLight(
            item_entity_ids=self.item_entity_ids,
            edge=self.edges,
            entity_description=self.rec_dataset.entity_description,
            use_description=self.use_description,
            n_entity=self.n_entity,
            n_relation=self.n_relation,
            n_base=self.n_base,
            vocab_size=self.vocab_size,
            use_context=self.use_context,
            text_pooling_type=self.text_pooling_type,
            text_pooling_head=self.text_pooling_head,
            kg_embedding_dim=self.kg_embedding_dim,
            learn_pos_embeds=self.learn_pos_embed,
            max_n_positions=self.max_n_positions,
            use_kg_loss=self.use_kg_loss,
            kg_margin=self.kg_margin,
            lr=lr,
            weight_decay=weight_decay,
            bart_lr=bart_lr,
            total_steps=self.rec_total_steps,
            warmup_steps=self.warmup_steps,
            num_cycles=self.num_cycles,
            phase=phase)

    def _init_lightning_conv(self):
        self.conv_model = KERLGenLight(
            rec_model=self.rec_model.rec_model,
            tokenizer=self.conv_dataset.tokenizer,
            lr=self.conv_lr,
            weight_decay=self.conv_weight_decay,
            total_steps=self.conv_total_steps,
            warmup_steps=self.warmup_steps,
            num_cycles=self.num_cycles,
            mx_resp_len=self.conv_dataset.max_response_length,
            kg_vocab_mask=self.conv_dataset.kg_vocab_mask,
            output_dir=self.gen_output_dir,
        )

        del self.rec_dataset

    def _load_best_model(self, model_path, phase):
        """
        Load the best model from the given path.
        """
        logger.info(f"best model path from {phase}: {model_path}")
        if phase in ["rec", "pretrain"]:
            self.rec_model = self.rec_model.load_from_checkpoint(
                checkpoint_path=model_path,
                item_entity_ids=self.item_entity_ids,
                edge=self.edges,
                entity_description=self.rec_dataset.entity_description,
                use_description=self.use_description,
                n_entity=self.n_entity,
                n_relation=self.n_relation,
                n_base=self.n_base,
                vocab_size=self.vocab_size,
                use_context=self.use_context,
                text_pooling_type=self.text_pooling_type,
                text_pooling_head=self.text_pooling_head,
                kg_embedding_dim=self.kg_embedding_dim,
                learn_pos_embeds=self.learn_pos_embed,
                max_n_positions=self.max_n_positions,
                use_kg_loss=self.use_kg_loss,
                kg_margin=self.kg_margin,
                lr=self.rec_lr,
                bart_lr=self.bart_lr,
                weight_decay=self.rec_weight_decay,
                num_cycles=self.num_cycles,
                total_steps=self.rec_total_steps,
                phase=phase)
        else:
            logger.debug("Load conv model")
            self.conv_model = self.conv_model.load_from_checkpoint(
                checkpoint_path=model_path,
                rec_model=self.rec_model.rec_model,
                tokenizer=self.conv_dataset.tokenizer,
                lr=self.conv_lr,
                weight_decay=self.conv_weight_decay,
                total_steps=self.conv_total_steps,
                warmup_steps=self.warmup_steps,
                num_cycles=self.num_cycles,
                mx_resp_len=self.conv_dataset.max_response_length,
                kg_vocab_mask=self.conv_dataset.kg_vocab_mask,
                output_dir=self.gen_output_dir,
            )

    def pretrain(self):
        logger.info("Start pretraining...")
        self.rec_dataset.change_stage("pretrain")
        self.rec_total_steps = self.rec_dataset.get_total_steps("kg") * self.pretrain_epoch
        self.warmup_steps = self.rec_total_steps // 10

        logger.info(f"Total number of edges in kg: {len(self.rec_dataset.edge_dataset)}")
        logger.info(f"Total number of steps in kg: {self.rec_total_steps}")
        logger.info(f"Total number of warmup steps in kg: {self.warmup_steps}")

        self._init_lightning_recommender(phase="pretrain")
        checkpoint = self.build_checkpoint_callback("pretrain")
        lr_monitor = LearningRateMonitor()
        rec_trainer = Trainer(accelerator="gpu",
                              devices=-1,
                              strategy=self.gpu_strategy,
                              auto_select_gpus=True,
                              max_epochs=self.pretrain_epoch,
                              logger=[self.logger],
                              check_val_every_n_epoch=1,
                              val_check_interval=1.0,
                              log_every_n_steps=50,
                              precision=16,
                              num_sanity_val_steps=0,
                              enable_progress_bar=True,
                              gradient_clip_val=self.gradient_clip_val,
                              # limit_train_batches=0.05,  # for debugging
                              deterministic=True,
                              callbacks=[checkpoint, lr_monitor])

        if not self.pretrain_reload:
            rec_trainer.fit(self.rec_model, self.rec_dataset)
        else:
            logger.info("Reload recommender model from specified path.")
            self._load_best_model(self.pretrain_reload_path, "pretrain")
        logger.info("Pretraining finished.")
        rec_trainer.test(self.rec_model, datamodule=self.rec_dataset)
        return self.pretrain_reload_path if self.pretrain_reload else checkpoint.last_model_path

    def train_recommender(self, pretrain_model_path=None):
        logger.info("Start training recommender...")
        logger.info(f"total train set: {len(self.rec_dataset.train_dataset)}")
        logger.info(f"total valid set: {len(self.rec_dataset.val_dataset)}")
        logger.info(f"total test set: {len(self.rec_dataset.test_dataset)}")
        self.rec_dataset.change_stage("rec")
        # 1 epoch for warmups
        self.warmup_steps = self.rec_dataset.get_total_steps("rec")
        self.rec_total_steps = self.warmup_steps * self.rec_epochs
        self.warmup_steps = self.rec_total_steps // 10

        logger.info(f"Recommendation warmup steps:{self.warmup_steps}")
        logger.info(f"Recommendation total steps:{self.rec_total_steps}")

        if not self.use_kg_loss:
            self._init_lightning_recommender("rec")
        else:
            self._load_best_model(pretrain_model_path, "rec")

        early_stop = self.build_early_stop_callback(train_stage='rec')
        checkpoint = self.build_checkpoint_callback(train_stage='rec')
        lr_monitor = LearningRateMonitor()
        if self.logger_type == "wandb":
            logger.warning("Using wandb logger.")
            self.logger.watch(self.rec_model.rec_model, log="gradients", log_freq=100)

        rec_trainer = Trainer(accelerator="gpu",
                              devices=-1,
                              strategy=self.gpu_strategy,
                              auto_select_gpus=True,
                              max_epochs=self.rec_epochs,
                              logger=[self.logger],
                              check_val_every_n_epoch=1,
                              # limit_val_batches=0,
                              val_check_interval=1.0,
                              log_every_n_steps=50,
                              precision=16,
                              num_sanity_val_steps=0,
                              deterministic=True,
                              enable_progress_bar=True,
                              # limit_train_batches=0.05,  # for debugging
                              gradient_clip_val=self.gradient_clip_val,
                              callbacks=[early_stop, checkpoint, lr_monitor])
        if not self.rec_reload:
            rec_trainer.fit(self.rec_model, self.rec_dataset)
            self._load_best_model(checkpoint.best_model_path, "rec")
        else:
            logger.info("Reload recommender model from specified path.")
            self._load_best_model(self.rec_model_path, "rec")

        logger.info("Finished training recommender, load best model")
        rec_trainer.test(self.rec_model, datamodule=self.rec_dataset)
        return self.rec_model_path if self.rec_reload else checkpoint.best_model_path

    def train_conversation(self):
        logger.info("Start training conversation module...")
        self.conv_dataset.change_stage("conv")
        logger.info(f"total train set: {len(self.conv_dataset.train_dataset)}")
        logger.info(f"total valid set: {len(self.conv_dataset.val_dataset)}")
        logger.info(f"total test set: {len(self.conv_dataset.test_dataset)}")
        self.warmup_steps = self.conv_dataset.get_total_steps("conv")
        self.conv_total_steps = self.warmup_steps * self.conv_epochs
        self.warmup_steps = self.conv_total_steps // 10

        logger.info(f"Conversation warmup steps:{self.warmup_steps}")
        logger.info(f"Conversation total steps:{self.conv_total_steps}")

        early_stop = self.build_early_stop_callback(train_stage='conv')
        checkpoint = self.build_checkpoint_callback(train_stage='conv')
        lr_monitor = LearningRateMonitor()
        self._init_lightning_conv()
        if self.logger_type == "wandb":
            logger.warning("Using wandb logger.")
            self.logger.watch(self.conv_model.generator, log="gradients", log_freq=100)

        gen_trainer = Trainer(accelerator="gpu",
                              devices=-1,
                              strategy=self.gpu_strategy,
                              auto_select_gpus=True,
                              max_epochs=self.conv_epochs,
                              logger=[self.logger],
                              check_val_every_n_epoch=1,
                              # limit_val_batches=0,
                              val_check_interval=1.0,
                              log_every_n_steps=50,
                              precision=16,
                              num_sanity_val_steps=0,
                              deterministic=True,
                              enable_progress_bar=True,
                              gradient_clip_val=self.gradient_clip_val,
                              # limit_train_batches=0.05,  # for debugging
                              callbacks=[early_stop, checkpoint, lr_monitor])

        if not self.conv_reload:
            gen_trainer.fit(self.conv_model, self.conv_dataset)
            self._load_best_model(checkpoint.best_model_path, "conv")
        else:
            logger.info("Reload conversation model from specified path.")
            self._load_best_model(self.conv_model_path, "conv")

        logger.info("Finished training conversation module, load best model")
        gen_trainer.test(self.conv_model, datamodule=self.conv_dataset)

    def fit(self):
        best_model_path = self.pretrain() if self.use_kg_loss else None
        self.train_recommender(best_model_path)
        self.train_conversation()
        out_dir = os.path.join(self.csv_log_path, self.log_name)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
            logger.warning(
                f"Remove {out_dir} to save disk space. You should not delete it if you want to keep the weights.")

        else:
            logger.warning(f"{out_dir} does not exist.")
