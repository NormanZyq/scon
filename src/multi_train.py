# This is a customized training script. Modified from train.py

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
import numpy as np
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torch.serialization import add_safe_globals
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.models.components.myconv import TwoLayerGNN
from src.models.components.gated_conv import GatedEncoder
from src.models.components.decoder import FeatureDecoder
from pygod.nn.decoder import DotProductDecoder
from torch_geometric.nn import GCNConv, GATConv, GraphConv, SAGEConv
from torch_geometric.nn.aggr.basic import SumAggregation

# Allow torch.load(weights_only=True) to trust our custom modules in checkpoints
add_safe_globals([
    TwoLayerGNN,
    GatedEncoder,
    FeatureDecoder,
    DotProductDecoder,
    GCNConv,
    GATConv,
    GraphConv,
    SAGEConv,
    SumAggregation,
])

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)        

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        # ckpt_path = trainer.checkpoint_callback.best_model_path
        # if ckpt_path == "":
        #     log.warning("Best ckpt not found! Using current weights for testing...")
        #     ckpt_path = None
        ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path skipped (using current weights).")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


def summarize(values, *, pct=True, ddof=0):
    arr = np.asarray(values, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=ddof)
    if pct:
        mean *= 100
        std *= 100
    return round(mean, 2), round(std, 2)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)

    seeds = list(range(10))

    legacy_groups = [
        ("test/o_auc", "test/attr_auc", "test/str_auc"),
        ("test/o_auc_init", "test/attr_auc_init", "test/str_auc_init"),
    ]
    merged_groups = [
        ("test/o_auc",),
        ("test/o_auc_init",),
    ]
    legacy_keys = [k for g in legacy_groups for k in g]
    merged_keys = [k for g in merged_groups for k in g]
    active_groups = None
    active_keys = None

    # Collect：dict[key] -> list[float]
    metrics = defaultdict(list)

    for seed in seeds:
        cfg_run = OmegaConf.merge(cfg, {"seed": seed})  
        metric_dict, _ = train(cfg_run)

        if active_keys is None:
            if all(k in metric_dict for k in legacy_keys):
                active_groups = legacy_groups
                active_keys = legacy_keys
            elif all(k in metric_dict for k in merged_keys):
                active_groups = merged_groups
                active_keys = merged_keys
            else:
                missing = [k for k in legacy_keys + merged_keys if k not in metric_dict]
                raise KeyError(f"Missing metric(s) in metric_dict keys={list(metric_dict.keys())}: {missing}")

        for k in active_keys:
            metrics[k].append(metric_dict[k])

    if active_groups is None:
        raise RuntimeError("No metrics collected; check training run for failures.")

    # Print once
    for i, seed in enumerate(seeds):
        flat_keys = [k for g in active_groups for k in g]
        if not all(k in metrics for k in flat_keys):
            continue
        parts = []
        for k in flat_keys:
            parts.append(f"{k.split('/')[-1]}: {metrics[k][i]:.4f}")
        print(f"seed {seed:5d}: " + ", ".join(parts))

    # Print by group
    for g in active_groups:
        if not all(k in metrics for k in g):
            continue
        lines = []
        for k in g:
            m, s = summarize(metrics[k], pct=True)
            name = k.split("/")[-1]
            lines.append(f"{name}: {m:.2f} (±{s:.2f})")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
