
import argparse

import torch

from models.classifier import DiseaseDetector


def build_model(args: argparse.Namespace, num_classes: int) -> list[torch.nn.Module]:
    return [DiseaseDetector(args.model, args.input_channels, num_classes) for _ in range(args.num_fold)]
