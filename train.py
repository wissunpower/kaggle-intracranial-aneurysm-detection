
import torch

from omegaconf import DictConfig
import hydra

from utils.log import logger
from utils.misc import fix_random_seed
from dataset import build_raw_data
from dataset.config import SeriesDataConfig
from models import build_model
from engine import build_trainer


def train(cfg: DictConfig):
    logger.info("Setting Configuration.. : ")
    logger.info(cfg)
    print("----------------------------------------------------------")

    device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')
    logger.info(f'device: {device}')

    if cfg.get("seed"):
        fix_random_seed(cfg.seed)

    data_config: SeriesDataConfig = hydra.utils.instantiate(cfg.data.config)

    models = build_model(cfg)

    hydra.utils.instantiate(cfg.monitor)

    for fold_index, model in enumerate(models):
        model = model.to(device)
        
        trainer = build_trainer(cfg, data_config, device, model, fold_index)
        
        trainer.train(model)
        
        # trainer.show_loss_and_epoch_graph()
        
        del trainer


@hydra.main(version_base="1.3", config_path="./_configs", config_name="train.yaml")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == '__main__':
    main()
