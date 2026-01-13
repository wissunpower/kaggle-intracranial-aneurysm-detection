
from omegaconf import DictConfig
import torch

from dataset import SeriesDataManager
from evaluator.dicom_series_evaluator import DicomSeriesEvaluator
from evaluator.vessel_segment_evaluator import VesselSegmentEvaluator
from evaluator.vessel_roi_bbox_evaluator import VesselROIBBoxEvaluator


def build_evaluator(cfg: DictConfig, data_config: SeriesDataManager
                    , device: torch.device|str, fold_index: int=0) -> DicomSeriesEvaluator:
    return DicomSeriesEvaluator(cfg, data_config, device, fold_index)
