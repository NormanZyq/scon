# Node Anomaly Module
import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from pygod.nn.decoder import DotProductDecoder
from torch_geometric.utils import to_dense_adj
from torchmetrics import AUROC, MaxMetric, MeanMetric

from .components.decoder import FeatureDecoder
from .components.f_xy import f_xy
# from .components.gated_conv import GatedEncoder
from .components.losses import recon_error
# from .components.myconv import TwoLayerGNN

def init_logits_from_error(
    logits_param,
    error_per_node,
    *,
    rho: float,
    low_p: float = 0.05,
    high_p: float = 0.95,
):
    n = int(error_per_node.numel())
    if n == 0:
        return
    k = max(1, int(round(n * float(rho))))
    k = min(k, n)

    idx = torch.topk(error_per_node, k=k, largest=True).indices
    low_p = min(max(float(low_p), 1e-4), 1.0 - 1e-4)
    high_p = min(max(float(high_p), 1e-4), 1.0 - 1e-4)
    low_logit = math.log(low_p / (1.0 - low_p))
    high_logit = math.log(high_p / (1.0 - high_p))

    init_logits = torch.full((n,), low_logit, device=logits_param.device, dtype=logits_param.dtype)
    init_logits[idx] = high_logit
    with torch.no_grad():
        logits_param.copy_(init_logits)


class NodeAnomalyModule(LightningModule):
    def __init__(
        self,
        num_nodes,
        warmup_encoder: nn.Module,
        encoder_gated: nn.Module,
        feature_decoder: FeatureDecoder,
        str_decoder: DotProductDecoder,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        warm_epochs=30,
        tau=1.0,
        gate_mode='mono',
        attr_err_weight=0.5,
        rho: float | None = None,
        # rho_attr: float | None = None,
        # rho_str: float | None = None,
        combine_o_method: str | None = None,
        reg_lambda=0.1,      # weight for entropy
        pos_weight_a=0.5,    # do not change, use 0.5 only
        pos_weight_s=0.5,    # do not change, use 0.5 only
        bce_s=False,
        use_lp_loss=False,
        num_pos_samples=0,
        neg_ratio=1.0
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # self.net = net
        self.warmup_epochs = warm_epochs
        self.tau = float(tau)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        self.gae_loss = recon_error
        self.attr_err_weight = attr_err_weight

        self.encoder_gated = encoder_gated

        # logits
        self.o_logits = nn.Parameter(torch.full((num_nodes,), -5.0))
        if self.warmup_epochs > 0:
            self.o_logits.requires_grad_(False)
        self._logits_initialized = self.warmup_epochs <= 0
        self._last_warmup_err = None

        self.warmup_encoder = warmup_encoder    # two-layer
        self.warmup_attr_decoder = feature_decoder
        self.warmup_str_decoder = str_decoder

        self.attr_decoder = feature_decoder.clone()     # Pretrained decoder is aborted
        self.str_decoder = str_decoder      # No use, we have removed str decoder in implementation

        # metric objects for calculating and averaging accuracy across batches
        self.train_auc_o = AUROC(task='binary')
        # No early stopping
        # self.val_auc = AUROC(task='binary')
        self.test_auc_o = AUROC(task='binary')

        self.auc_init = AUROC(task='binary')

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.auc_best = MaxMetric()

        # buffers for test-time TSNE visualization
        self._test_z = []
        self._test_y = []
        self._test_y_available = True
        self._test_edge_index = None
        self._test_node_labels = None

    @property
    def o(self):
        return torch.sigmoid(self.o_logits / max(self.tau, 1e-6))

    @property
    def o_attr(self):
        return self.o

    @property
    def o_str(self):
        return self.o

    @staticmethod
    def _binary_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        p = p.clamp(min=eps, max=1.0 - eps)
        return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))

    def _merge_errors(self, attr_err: torch.Tensor, str_err: torch.Tensor) -> torch.Tensor:
        return self.attr_err_weight * attr_err + (1.0 - self.attr_err_weight) * str_err

    def _resolve_rho(self) -> float:
        rho = getattr(self.hparams, "rho", None)
        if rho is not None:
            return float(rho)
        return 0.1

    def warmup_forward(self, x, edge_index):
        z = self.warmup_encoder(x, edge_index)
        x_hat = self.warmup_attr_decoder(z)
        s_hat = self.warmup_str_decoder(z, edge_index)

        return z, x_hat, s_hat

    def _recon_errors(self, x, x_hat, s, s_hat, z, edge_index):
        attr_err, str_err = self.gae_loss(
            x,
            x_hat,
            s,
            s_hat,
            pos_weight_a=self.hparams.pos_weight_a,     
            pos_weight_s=self.hparams.pos_weight_s,
            bce_s=self.hparams.bce_s,
            use_lp_loss=self.hparams.use_lp_loss,
            # use_lp_loss=(self.hparams.use_lp_loss and self.current_epoch >= self.warmup_epochs),
            pos_edge_index=edge_index,
            z=z,
            num_pos_samples=self.hparams.num_pos_samples,
            neg_ratio=self.hparams.neg_ratio,
        )
        if isinstance(str_err, tuple):
            str_err, _ = str_err
        return attr_err, str_err

    def downstream_forward(self, x, edge_index) -> torch.Tensor:
        o = self.o
        z = self.encoder_gated(x, edge_index, o)
        x_hat = self.attr_decoder(z)
        s_hat = self.str_decoder(z, edge_index)
        # return z, x_hat, s_hat, o
        return z, x_hat, s_hat
    
    def forward(self, x, edge_index):
        return self.downstream_forward(x, edge_index)

    def warmup_model_step(self, batch):
        x, edge_index = batch.x, batch.edge_index
        batch_index = getattr(batch, "batch", None)
        s = to_dense_adj(edge_index, batch=batch_index)
        if s.dim() == 3 and s.size(0) == 1:
            s = s.squeeze(0)
        z, x_hat, s_hat = self.warmup_forward(x, edge_index)
        attr_err, str_err = self._recon_errors(x, x_hat, s, s_hat, z, edge_index)
        return attr_err, str_err

    def downstream_model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, edge_index = batch.x, batch.edge_index
        batch_index = getattr(batch, "batch", None)
        s = to_dense_adj(edge_index, batch=batch_index)
        if s.dim() == 3 and s.size(0) == 1:
            s = s.squeeze(0)
        z, x_hat, s_hat = self.downstream_forward(x, edge_index)
        attr_err, str_err = self._recon_errors(x, x_hat, s, s_hat, z, edge_index)
        loss_recon = self._merge_errors(attr_err, str_err)
        loss_recon = loss_recon.mean()
        o = self.o
        entropy_reg = self._binary_entropy(o).mean()
        loss = loss_recon + self.hparams.reg_lambda * entropy_reg
        return loss

    def model_step(self, batch, batch_idx):
        if self.current_epoch < self.warmup_epochs:
            # pretrain: Initialization phase
            attr_err, str_err = self.warmup_model_step(batch)
            merged_err = self._merge_errors(attr_err, str_err)
            self._last_warmup_err = merged_err.detach()
            loss = merged_err.mean()
        else:
            # downstream: score-conditioned propagation phase
            if self.warmup_epochs > 0 and not self._logits_initialized:
                if self._last_warmup_err is None:
                    raise RuntimeError("Warmup errors are missing for logits initialization.")
                init_logits_from_error(self.o_logits, self._last_warmup_err, rho=self._resolve_rho())
                self.o_logits.requires_grad_(True)
                self._logits_initialized = True
                print("Logits initialized.")

                # Compute init auc                
                y = (batch.y > 0).long()
                self.auc_init(self.o.detach(), y)
                print('[auc_init]\no_auc={:.4f}'.format(self.auc_init.compute()))
            loss = self.downstream_model_step(batch)
        return loss
            
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.val_loss.reset()
        # self.auc.reset()
        pass

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        targets_raw = batch.y
        overall_target = (targets_raw > 0).long()

        loss = self.model_step(batch, batch_idx)
        o = self.o.detach()
        # update and log metrics
        self.train_loss(loss)
        self.train_auc_o(o, overall_target)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/o_auc", self.train_auc_o, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        auc = self.train_auc_o.compute()
        self.auc_best(auc)
        self.log('train/auc_best', self.auc_best.compute(), sync_dist=True, prog_bar=True)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass
        # loss, preds, targets = self.model_step(batch)

        # # update and log metrics
        # self.val_loss(loss)
        # self.val_acc(preds, targets)
        # self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        pass

    def on_test_start(self) -> None:
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        targets_raw = batch.y
        overall_target = (targets_raw > 0).long()

        loss = self.model_step(batch, batch_idx)
        o = self.o.detach()
        # update and log metrics
        self.test_loss(loss)
        self.test_auc_o(o, overall_target)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/o_auc", self.test_auc_o, on_step=False, on_epoch=True, prog_bar=True)

        # Also print the AUC immediately after the initialization
        self.log('test/o_auc_init', self.auc_init, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.warmup_encoder = torch.compile(self.warmup_encoder)
            self.encoder_gated = torch.compile(self.encoder_gated)
            self.warmup_attr_decoder = torch.compile(self.warmup_attr_decoder)
            self.attr_decoder = torch.compile(self.attr_decoder)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
