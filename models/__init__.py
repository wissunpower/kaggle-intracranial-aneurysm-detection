
from omegaconf import DictConfig
import hydra

import torch

from models.backbone import build_backbone
from models.vessel_segment_extractor import VesselSegmentExtractor
from models.vessel_roi_bbox_extractor import VesselROIBBoxExtractor


def build_model(cfg: DictConfig) -> list[torch.nn.Module]:
    return [hydra.utils.instantiate(cfg.model.backbone) for _ in range(cfg.data.num_fold)]
