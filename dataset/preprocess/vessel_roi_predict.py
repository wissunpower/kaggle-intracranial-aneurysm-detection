
import rootutils
import torch

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
import glob
from tqdm.auto import tqdm

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

EVAL_RESOLVER_NAME = "eval"
if not OmegaConf.has_resolver(EVAL_RESOLVER_NAME):
    OmegaConf.register_new_resolver(EVAL_RESOLVER_NAME, eval)

LEN_RESOLVER_NAME = "len"
if not OmegaConf.has_resolver(LEN_RESOLVER_NAME):
    OmegaConf.register_new_resolver(LEN_RESOLVER_NAME, len)

from utils.log import logger
from utils.misc import fix_random_seed


class VesselROIBBoxPredictor:
    def __init__(self
                 , data_root_path: str
                 , input_data_path: str
                 , batch_size: int
                 , load_model_path: str
                 , result_npy_file_name: str
                 , device: torch.device
                 , model: torch.nn.Module
                 ):
        self.data_root_path = data_root_path
        files = np.array(glob.glob(input_data_path + '*.npy'))
        self.metadata_df \
            = pd.DataFrame({
                'SeriesUID': [x.split('/')[-1].split('\\')[-1].split('_')[0] for x in files],
                'window_file': files,
                'x1': 0,
                'y1': 0,
                'x2': 0,
                'y2': 0,
                })
        self.batch_size = batch_size

        self.load_model_path = load_model_path
        self.result_npy_file_name = result_npy_file_name
        self.device = device
        self.model = model

    def __call__(self, valid_dataset):
        state_dict = torch.load(self.load_model_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)

        self.model.eval()

        valid_dataset.initialize(self.metadata_df)
        valid_dataloader = \
            torch.utils.data.DataLoader(valid_dataset, self.batch_size)
        
        OUTPUTS, IDS = [], []

        with torch.no_grad():
            for batch_index, batch_data in enumerate(tqdm(valid_dataloader)):
                volumes, _, ids = batch_data
                volumes = volumes.to(self.device)
                
                with torch.autocast(device_type=str(self.device)):
                    logits = self.model(volumes)
                
                OUTPUTS.extend(logits.float().detach().cpu().numpy())
                IDS.extend(ids)
                
                # if batch_index > 2:
                #     break
        
        OUTPUTS = np.stack(OUTPUTS)
        IDS = np.stack(IDS)

        print(np.mean(OUTPUTS[:, 1] - OUTPUTS[:, 0]))
        print(np.mean(OUTPUTS[:, 3] - OUTPUTS[:, 2]))

        SID_TO_PRED = {iid.split('/')[-1].split('\\')[-1].split('_')[0]: output
                       for iid, output in zip(IDS, OUTPUTS)}
        np.save(self.data_root_path + self.result_npy_file_name + '.npy', SID_TO_PRED)
        
        print("Predict complete.")


def predict(cfg: DictConfig):
    logger.info("Setting Configuration.. : ")
    logger.info(cfg)
    print("----------------------------------------------------------")

    device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')
    logger.info(f'device: {device}')

    if cfg.get("seed"):
        fix_random_seed(cfg.seed)

    valid_dataset = hydra.utils.instantiate(cfg.data.vessel_roi_bbox.valid_dataset)

    model = hydra.utils.instantiate(cfg.model.vessel_roi_bbox)

    predictor: VesselROIBBoxPredictor \
        = hydra.utils.instantiate(cfg.predict
                                  , device=device
                                  , model=model)

    predictor(valid_dataset)


@hydra.main(version_base="1.3", config_path="../../_configs", config_name="vessel_roi_predict.yaml")
def main(cfg: DictConfig):
    predict(cfg)


if __name__ == '__main__':
    main()
