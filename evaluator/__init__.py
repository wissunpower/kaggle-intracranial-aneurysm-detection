
from omegaconf import DictConfig
import torch

from dataset.config import SeriesDataConfig
from evaluator.dicom_series_evaluator import DicomSeriesEvaluator


def build_evaluator(cfg: DictConfig, data_config: SeriesDataConfig
                    , device: torch.device|str, fold_index: int=0) -> DicomSeriesEvaluator:
    return DicomSeriesEvaluator(cfg, data_config, device, fold_index)
