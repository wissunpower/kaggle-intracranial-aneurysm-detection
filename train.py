
import torch

from utils.misc import parse_args, fix_random_seed
from dataset import build_raw_data
from dataset.config import NUM_CLASSES, SeriesDataConfig
from models import build_model
from engine import build_trainer


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')
    print(f'device: {device}')

    fix_random_seed(args)
    
    build_raw_data(args)

    data_config = SeriesDataConfig(args)

    models = build_model(args, NUM_CLASSES)

    for fold_index, model in enumerate(models):
        model = model.to(device)
        
        trainer = build_trainer(args, data_config, device, model, fold_index)
        
        trainer.train(model)
        
        # trainer.show_loss_and_epoch_graph()
        
        del trainer


if __name__ == '__main__':
    train()
