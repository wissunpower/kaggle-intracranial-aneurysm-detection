
from omegaconf import DictConfig
import hydra

import torch


def build_model(cfg: DictConfig) -> list[torch.nn.Module]:
    return [hydra.utils.instantiate(cfg.model.backbone) for _ in range(cfg.data.num_fold)]
