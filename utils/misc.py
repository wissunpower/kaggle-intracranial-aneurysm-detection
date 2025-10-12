
import os, random
import numpy as np
import argparse

import torch
import torchvision


def parse_args(arg_params: list[str]|None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Real-time Object Detection LAB')
    # Random seed
    parser.add_argument('--seed', default=42, type=int)

    # GPU
    # parser.add_argument('--cuda', action='store_true', default=False, help='use cuda.')
    
    # Image size
    parser.add_argument('-size', '--img_size', default=416, type=int, help='input image size')
    parser.add_argument('--img_crop_size', default=224, type=int, help='input image crop size')
    parser.add_argument('--eval_first', action='store_true', default=False, help='evaluate model before training.')
    
    # Outputs
    parser.add_argument('--tfboard', action='store_true', default=False, help='use tensorboard')
    parser.add_argument('--save_folder', default='./checkpoints/', type=str, help='path to save weight')
    parser.add_argument('--vis_tgt', action="store_true", default=False, help="visualize training data.")
    parser.add_argument('--vis_aux_loss', action="store_true", default=False, help="visualize aux loss.")
    
    # Mixing precision
    parser.add_argument('--fp16', dest="fp16", action="store_true", default=False, help="Adopting mix precision training.")
    
    # Batchsize
    parser.add_argument('-bs', '--batch_size', default=16, type=int, help='batch size on all the GPUs.')

    # Epoch
    parser.add_argument('--max_epoch', default=10, type=int, help='max epoch.')
    parser.add_argument('--wp_epoch', default=1, type=int, help='warmup epoch.')
    parser.add_argument('--eval_epoch', default=10, type=int, help='after eval epoch, the model is evaluated on val dataset.')
    parser.add_argument('--no_aug_epoch', default=20, type=int, help='cancel strong augmentation.')

    # Model
    parser.add_argument('-m', '--model', default=torchvision.models.resnet18.__name__.lower(), type=str, help='select model')
    parser.add_argument('-ct', '--conf_thresh', default=0.005, type=float, help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.6, type=float, help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int, help='topk candidates dets of each level before NMS')
    parser.add_argument('-p', '--pretrained', default=None, type=str, help='load pretrained weight')
    parser.add_argument('-r', '--resume', default=None, type=str, help='keep training')
    parser.add_argument('--nms_class_agnostic', action='store_true', default=False, help='Perform NMS operations regardless of category.')
    parser.add_argument('--num_fold', default=1, type=int, help='number of K-fold')
    parser.add_argument('--input_channels', default=32, type=int, help='number of model input channels')
    parser.add_argument('--model_weight_path', default=None, type=str, help='Trained state_dict file path to open')

    # Dataset
    parser.add_argument('--data_path', default='F:/ml_data_resource/kaggle/intracranial_aneurysm_detection/', help='data root')
    parser.add_argument('-d', '--dataset', default='voc', help='coco, voc, widerface, crowdhuman')
    parser.add_argument('--load_cache', action='store_true', default=False, help='Path to the cached data.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--label_file_name', default='train.csv', type=str, help='data label file name')
    parser.add_argument('--label_weight_priority', default=[], type=list[str], help='weight priority of disease label column')
    parser.add_argument('--preprocess_data_folder_name', default='raw_data/', type=str, help='preprocess data(npz files) folder name')
    
    # Train trick
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False, help='Multi scale')
    parser.add_argument('--ema', action='store_true', default=False, help='Model EMA')
    parser.add_argument('--min_box_size', default=8.0, type=float, help='min size of target bounding box.')
    parser.add_argument('--mosaic', default=None, type=float, help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float, help='mixup augmentation.')
    parser.add_argument('--grad_accumulate', default=1, type=int, help='gradient accumulation')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False, help='distributed training')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, help='use sybn.')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode.')

    args = parser.parse_args(arg_params)

    if not args.data_path.endswith(os.path.sep) \
        and not args.data_path.endswith('/'):
        # Ensure output dir is valid for later use
        args.data_path += os.path.sep

    if not args.preprocess_data_folder_name.endswith(os.path.sep) \
        and not args.preprocess_data_folder_name.endswith('/'):
        # Ensure output dir is valid for later use
        args.preprocess_data_folder_name += os.path.sep

    if not args.save_folder.endswith(os.path.sep) \
        and not args.save_folder.endswith('/'):
        # Ensure output dir is valid for later use
        args.save_folder += os.path.sep

    return args

def fix_random_seed(args: argparse.Namespace):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
