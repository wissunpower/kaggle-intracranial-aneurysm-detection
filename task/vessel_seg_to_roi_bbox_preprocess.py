
import rootutils

from omegaconf import DictConfig, OmegaConf
import hydra

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

EVAL_RESOLVER_NAME = "eval"
if not OmegaConf.has_resolver(EVAL_RESOLVER_NAME):
    OmegaConf.register_new_resolver(EVAL_RESOLVER_NAME, eval)

from utils.log import logger
from dataset.preprocess import VesselSegToROIBBoxPreprocessor


def preprocess(cfg: DictConfig):
    logger.info("Setting Configuration.. : ")
    logger.info(cfg)
    print("----------------------------------------------------------")

    preprocessor: VesselSegToROIBBoxPreprocessor = hydra.utils.instantiate(cfg.preprocess)

    preprocessor()


@hydra.main(version_base="1.3", config_path="../_configs", config_name="vessel_seg_to_roi_bbox_preprocess.yaml")
def main(cfg: DictConfig):
    preprocess(cfg)


if __name__ == '__main__':
    main()
