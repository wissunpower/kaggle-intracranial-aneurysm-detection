
import rootutils
import torch

from omegaconf import DictConfig, OmegaConf
import hydra

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

EVAL_RESOLVER_NAME = "eval"
if not OmegaConf.has_resolver(EVAL_RESOLVER_NAME):
    OmegaConf.register_new_resolver(EVAL_RESOLVER_NAME, eval)

LEN_RESOLVER_NAME = "len"
if not OmegaConf.has_resolver(LEN_RESOLVER_NAME):
    OmegaConf.register_new_resolver(LEN_RESOLVER_NAME, len)

from utils.log import logger
from utils.misc import fix_random_seed
from dataset import VesselSegmentDataManager
from trainer import VesselSegmentTrainer


def train(cfg: DictConfig):
    logger.info("Setting Configuration.. : ")
    logger.info(cfg)
    print("----------------------------------------------------------")

    device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')
    logger.info(f'device: {device}')

    if cfg.get("seed"):
        fix_random_seed(cfg.seed)

    data_manager: VesselSegmentDataManager \
        = hydra.utils.instantiate(cfg.data.vessel_segment, data_common_cfg=cfg.data.common)
    data_manager.prepare_data()

    model = hydra.utils.instantiate(cfg.model.vessel_segment)

    hydra.utils.instantiate(cfg.monitor)
    
    trainer: VesselSegmentTrainer \
        = hydra.utils.instantiate(cfg.trainer.vessel_segment
                                  , device=device
                                  , data_manager=data_manager
                                  , model=model
                                  )

    trainer.train()


@hydra.main(version_base="1.3", config_path="../_configs", config_name="vessel_segment_train.yaml")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == '__main__':
    main()
