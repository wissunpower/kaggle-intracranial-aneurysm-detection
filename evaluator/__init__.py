
import argparse
import torch

from dataset.config import SeriesDataConfig
from evaluator.dicom_series_evaluator import DicomSeriesEvaluator


def build_evaluator(args: argparse.Namespace, data_config: SeriesDataConfig
                    , device: torch.device|str, fold_index: int=0) -> object:
    return DicomSeriesEvaluator(args, data_config, device, fold_index)
