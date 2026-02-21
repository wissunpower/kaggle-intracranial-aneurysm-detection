import os
import numpy as np
import pandas as pd
from collections import defaultdict
from pprint import pprint

from sklearn.metrics import roc_auc_score

import rootutils

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


def select_predict(cfg: DictConfig):
    logger.info("Setting Configuration.. : ")
    logger.info(cfg)
    print("----------------------------------------------------------")
    
    train_sampled_df = pd.read_csv(cfg.slide_metainfo_file_path)
    train_sampled_df['image_file'] = train_sampled_df.apply(lambda row: f'{row.SeriesUID}_I_{row.InstanceNumber}', axis=1)
    filename_to_index = dict(zip(train_sampled_df.image_file, train_sampled_df.index))
    
    label_cols = train_sampled_df.columns[6:6+13]

    predicts, labels, uids = [], [], []
    
    for fold_index in range(cfg.num_fold):
        fold_predicts = np.load(os.path.join(cfg.fold_result_root_paths[fold_index], f'predict_{fold_index:02d}.npy'))
        fold_labels = np.load(os.path.join(cfg.fold_result_root_paths[fold_index], f'label_{fold_index:02d}.npy'))
        fold_uids = np.load(os.path.join(cfg.fold_result_root_paths[fold_index], f'uid_{fold_index:02d}.npy'))
        
        predicts.extend(fold_predicts)
        labels.extend(fold_labels)
        uids.extend(fold_uids)
    
    predicts, labels, uids = np.array(predicts), np.array(labels), np.array(uids)
    
    
    series_groups = np.array([uid.split('_')[0] for uid in uids])
    group_indices = defaultdict(list)
    for slide_index, group_name in enumerate(series_groups):
        group_indices[group_name].append(slide_index)
    
    
    series_predicts, series_labels, series_uids = [], [], []
    
    for series_uid in np.unique(series_groups):
        series_predicts.append(predicts[group_indices[series_uid]].max(0))
        series_labels.append(labels[group_indices[series_uid]].max(0))
        series_uids.append(series_uid)
    
    series_predicts, series_labels, series_uids = np.array(series_predicts), np.array(series_labels), np.array(series_uids)
    
    
    classes_to_score = {}
    scores = []
    for index, class_name in enumerate(label_cols.tolist() + ['aneurysm_present']):
        score = roc_auc_score(series_labels[:, index], series_predicts[:, index])
        classes_to_score[class_name] = score
        scores.append(score)
    
    total_score = (np.mean(scores[:len(scores) - 1]) + scores[-1]) / 2
    
    print(f'Patient-Level AUC: {total_score}')
    pprint(classes_to_score)
    
    
    # positive_predicts_df = pd.DataFrame(predicts[labels[:, -1] == 1.])
    
    # positive_predicts_df.describe()
    
    # desc_percentiles = np.arange(start=95, stop=100, step=1) / 100
    # positive_predicts_df.describe(percentiles=desc_percentiles.tolist())
    
    positive_indices = []

    if cfg.use_soft_target:
        positive_uids = uids[labels[:, -1] == 1.]
        
        for positive_uid in positive_uids:
            row_index = filename_to_index[positive_uid]
            positive_indices.append(row_index)
        
        positive_data_df = train_sampled_df.iloc[positive_indices].reset_index(drop=True)
        positive_data_df['soft_target'] = [predict.tolist() for predict in predicts[labels[:, -1] == 1.]]
    else:
        predict_sum_scores = predicts[labels[:, -1] == 1.].sum(1)
        positive_uids = uids[labels[:, -1] == 1.][predict_sum_scores > cfg.positive_threshold]
        
        positive_data_rows = []
    
        for positive_uid in positive_uids:
            row_index = filename_to_index[positive_uid]
            positive_indices.append(row_index)
            positive_data_rows.append(train_sampled_df.iloc[row_index])
        
        positive_data_df = pd.DataFrame(positive_data_rows)
    
    positive_data_df['total_index'] = positive_indices
    
    positive_data_df.to_csv(cfg.positive_data_output_path, index=False)


@hydra.main(version_base="1.3", config_path="../_configs", config_name="classify_predict_select.yaml")
def main(cfg: DictConfig):
    select_predict(cfg)


if __name__ == '__main__':
    main()
