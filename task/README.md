## 실행 순서 안내
### 전체 흐름 요약
1. 데이터 전처리
    1. series 데이터를 개별 슬라이드 단위로 분리
        1. ./dataset/preprocess/dcm_to_npy.py
    2. 혈관 ROI 추출 단계의 학습을 위한 label(bounding box 형식) 생성과 예측을 위한 모든 series 데이터의 전처리
        1. ./task/vessel_seg_to_roi_bbox_preprocess.py

2. 혈관 ROI 추출 모델 훈련 : Stage 1
    1. 혈관 ROI bounding box를 예측하도록 모델을 훈련
        1. ./task/vessel_roi_bbox_train.py
    2. 모든 series 데이터에 대한 혈관 ROI bounding box 추출
        1. ./dataset/preprocess/vessel_roi_predict.py

3. 뇌동맥류 질환 존재 여부 및 위치에 대한 다중 이진 분류 훈련 : Stage 2
    1. (선택) 분류 학습 데이터인 개별 슬라이드 이미지 crop 전처리
        1. ./dataset/preprocess/dcm_to_npy.py
    2. 개별 슬라이드 단위의 데이터를 바탕으로 **이진 분류 모델 훈련**
        1. ./task/classifier_train.py
    3. 훈련된 분류 모델을 바탕으로 positive 슬라이드 데이터 증강
        1. ./task/classify_predict_select.py
    4. positive 슬라이드를 증강한 데이터에 추가하여 **이진 분류 모델 훈련**
        1. ./task/classifier_train.py
    5. 지식 증류(Knowledge Distillation) 방식의 soft target 정보 추출 후 데이터 증강
        1. ./task/classify_predict_select.py"
    6. soft target 이 포함된 positive 슬라이드를 학습 데이터에 추가하여 **이진 분류 모델 훈련**
        1. ./task/classifier_train.py

4. 추론
    1. 실제 의학 영상 자료에 대한 모델 사용 사례
        1. ./task/for_submission.py

### 세부 설명
./task 폴더(현재 위치)와 ./dataset/preprocess 폴더에 있는 python 스크립트를 바탕으로 전체 작업을 실행하는 세부 방법은 아래와 같습니다.
**주요 설정** 내용은 각 단계를 실행하는데 필요한 입력 데이터와 그에 따른 결과물과 직접적으로 관련된 사항을 주로 다루고 있습니다.
신경망 모델 종류와 이와 관련된 optimizer, 학습률 scheduler 같은 설정은 실행 의도에 따라 자유롭게 변경 가능하여 자세한 설명은 생략하였습니다.

1. 데이터 전처리
    1. 4348개의 series 데이터를 개별 슬라이드 단위로 분리하여 시각적 이미지 정보는 npy 형식으로 저장하고, 모든 슬라이드 색인 정보를 각각의 row 단위로 매핑한 csv 파일을 생성합니다.
        1. 실행 방법 : "python [./dataset/preprocess/dcm_to_npy.py](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/dataset/preprocess/dcm_to_npy.py)"
        2. 주요 설정 ("[./_configs/dcm_to_npy.yaml](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/_configs/dcm_to_npy.yaml)" 파일 참고)
            1. label_file_path
                1. 기본값 : ${paths.data_root_dir}train.csv
                2. kaggle 에서 제공한 series 의 기본 정보를 담고 있는 파일의 경로를 설정하면 됩니다. ${paths.data_root_dir}는 개발 환경에 따라 임의로 설정 가능합니다.
            2. series_data_path
                1. 기본값 : ${paths.data_root_dir}series/
                2. kaggle 에서 제공한 DICOM series 데이터 집합이 위치한 경로를 설정하면 됩니다.
            3. output_path
                1. 기본값 : ${paths.data_root_dir}raw_slide_data/
                2. series_data_path 에 있는 데이터를 개별 슬라이드 단위로 분리한 시각적 이미지 정보(npy 형식으로 생성한 파일)가 저장되는 위치입니다.
            4. slide_metainfo_gen
                1. enable
                    1. 기본값 : **true**
                    2. 개별 슬라이드 단위로 분할된 이미지 파일의 색인 정보를 담고 있는 파일을 생성하는 switch 입니다.
                2. output_path
                    1. 기본값 : ${paths.data_root_dir}train_sampled_new.csv
                    2. 개별 슬라이드 단위로 분할된 이미지 파일의 색인 정보를 담고 있는 파일이 저장되는 위치입니다.
            5. preprocess
                1. crop
                    1. enable
                        1. 기본값 : **false**
                        2. 개별 슬라이드 단위로 분할된 이미지를 저장할 때 crop 기능을 적용하는 switch 입니다.
                2. resize
                    1. enable
                        1. 기본값 : false
                        2. 개별 슬라이드 단위로 분할된 이미지를 저장할 때 resize 기능을 적용하는 switch 입니다.
    2. segmentations/ 안에 혈관 segment 정보가 있는 series 데이터에 한하여 혈관 ROI label(bounding box 형식)을 생성합니다. 또한 모든 4348건의 series 데이터에 대한 슬라이드 이미지 정보를 ROI 영역 추출의 입력 데이터로 사용할 수 있게 전처리하여 저장합니다.
        1. 실행 방법 : "python [./task/vessel_seg_to_roi_bbox_preprocess.py](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/task/vessel_seg_to_roi_bbox_preprocess.py)"
        2. 주요 설정 ("[./_configs/vessel_seg_to_roi_bbox_preprocess.yaml](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/_configs/vessel_seg_to_roi_bbox_preprocess.yaml)" 파일 참고)
            1. preprocess
                1. segment_folder_name
                    1. 기본값 : segmentations
                    2. kaggle 에서 제공한 178 건의 혈관 segment 정보가 담겨있는 segmentations/ 폴더 경로를 설정합니다.
                2. seg_output_name
                    1. 기본값 : segmentations_window1
                    2. 혈관 ROI를 추출하는 모델을 훈련하는데 사용할 segment 정보 npy 파일이 저장되는 폴더 이름입니다.
                3. total_output_name
                    1. 기본값 : train_windows1_test1_
                    2. 혈관 ROI 추출 모델을 사용하여 모든 series 데이터의 bounding box 예측을 할 때 입력되는 데이터의 저장 폴더 이름입니다.
                4. series_slide_fileset_path
                    1. 기본값 : ${paths.data_root_dir}raw_slide_data/
                    2. 개별 슬라이드에 대한 이미지가 저장된 위치 입니다. 1.i.b.c(dcm_to_npy.output_path) 설정과 동일한 대상입니다.
                5. series_slide_metainfo_file_name
                    1. 기본값 : train_sampled_new.csv
                    2. 개별 슬라이드 이미지 파일의 색인 정보를 저장한 파일 이름입니다. 1.i.b.d.b(dcm_to_npy.slide_metainfo_gen.output_path) 설정의 파일 이름과 같습니다.
                6. num_series_slide
                    1. 기본값 : 48
                    2. 각각의 series 데이터는 서로 다른 개수의 dicom 슬라이드로 구성되어 있는데, 전처리하는 과정에서 이는 같은 개수의 등간격 슬라이드로 조정됩니다.
                    등간격 처리되는 슬라이드의 개수를 나타냅니다.

2. 혈관 ROI 추출 모델 훈련 : Stage 1
    1. 전처리된 segmentations/ 안의 혈관 segment 데이터로 **혈관 ROI bounding box를 예측하도록 모델을 훈련**시킵니다.
        1. 실행 방법 : "python [./task/vessel_roi_bbox_train.py](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/task/vessel_roi_bbox_train.py)"
        2. 주요 설정
            1. data.vessel_roi_bbox ([./_configs/data/default.yaml](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/_configs/data/default.yaml) 파일 참고)
                1. metadata_file_path
                    1. 기본값 : ${paths.data_root_dir}segmentations_window1.csv
                    2. 모델 훈련에 사용하는 segment 정보 npy 파일의 색인 정보가 있는 파일입니다. 1.ii.b.a.b(vessel_seg_to_roi_bbox_preprocess.preprocess.seg_output_name) 설정의 폴더와 같은 이름의 csv 파일입니다.
            2. model.vessel_roi_bbox ([./_configs/model/default.yaml](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/_configs/model/default.yaml) 파일 참고)
                1. backbone
                    1. kwargs
                        1. model_name
                            1. 기본값 : vit_small_plus_patch16_dinov3.lvd1689m
                            2. backbone 모델의 종류입니다.
            3. trainer.vessel_roi_bbox ([./_configs/trainer/default.yaml](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/_configs/trainer/default.yaml) 파일 참고)
                1. save_path
                    1. 기본값 : ${paths.save_folder}
                    2. 훈련된 모델의 가중치가 저장되는 위치입니다.
    2. 2.i 에서 훈련된 모델을 사용하여 4348건의 모든 series 데이터에 대한 혈관 ROI bounding box 추출을 진행합니다.
        1. 실행 방법 : "python [./dataset/preprocess/vessel_roi_predict.py](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/dataset/preprocess/vessel_roi_predict.py)"
        2. 주요 설정 ("[./_configs/vessel_roi_predict.yaml](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/_configs/vessel_roi_predict.yaml)" 파일 참고)
            1. predict
                1. input_data_path
                    1. 기본값 : ${predict.data_root_path}train_windows1_test1_/
                    2. 1.ii 과정에서 전처리된 series 단위의 슬라이드 집합입니다. 4348 건 모두 같은 개수의 등간격 슬라이드로 재조정되었으며, ROI 추출을 위한 입력 데이터로 사용됩니다. 1.ii.b.a.c(vessel_seg_to_roi_bbox_preprocess.preprocess.total_output_name) 과 동일한 대상을 가리킵니다.
                2. load_model_path
                    1. 기본값 : ./_results/00_02_09_apply_cropped_reference_setting/checkpoints/vesselroibboxextractor_20260129_153709/best_loss_checkpoint_vesselroibboxextractor_00_20260129_153709.pth **(예시)**
                    2. ROI 추출을 위해 훈련된 모델의 가중치가 있는 위치입니다. 2.i.b.c.a(trainer.vessel_roi_bbox.save_path) 을 참고하면 됩니다.
                3. result_npy_file_name
                    1. 기본값 : series_crop_bbox_loss_20260129_153709 **(예시)**
                    2. 예측된 ROI bounding box 정보는 x1, x2, y1, y2 형식(0.0 ~ 1.0 사이의 값)으로 npy 파일에 저장됩니다. 저장될 파일의 이름을 지정합니다.

3. 뇌동맥류 질환 존재 여부 및 위치에 대한 다중 이진 분류 훈련 : Stage 2
    1. (선택) 본격적인 분류 학습에 앞서 입력 데이터로 사용될 개별 슬라이드 이미지를 Stage 1에서 예측한 결과를 바탕으로 crop 처리를 합니다. 510GB가 넘는 원본 데이터의 용량을 150GB 이하로 줄일 수 있으며, 이는 학습 속도에 많은 영향을 미치는 디스크 IO 부하를 낮춰 훈련 시간을 크게 줄일 수 있습니다. 해당 단계를 생략할 경우 분류 훈련 과정(3.2)에서 원본 데이터를 읽은 후 crop 전처리를 하여 동일한 효과를 얻을 수 있습니다.
        1. 실행 방법 : "python [./dataset/preprocess/dcm_to_npy.py](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/dataset/preprocess/dcm_to_npy.py)"
        2. 주요 설정 ("[./_configs/dcm_to_npy.yaml](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/_configs/dcm_to_npy.yaml)" 파일 참고)
            1. label_file_path
                1. 기본값 : ${paths.data_root_dir}train.csv
                2. kaggle 에서 제공한 series 의 기본 정보를 담고 있는 파일의 경로를 설정하면 됩니다. ${paths.data_root_dir}는 개발 환경에 따라 임의로 설정 가능합니다.
            2. series_data_path
                1. 기본값 : ${paths.data_root_dir}series/
                2. kaggle 에서 제공한 DICOM series 데이터 집합이 위치한 경로를 설정하면 됩니다.
            3. output_path
                1. 기본값 : ${paths.data_root_dir}raw_slide_data_cropped_loss_20260129_153709/ **(예시)**
                2. series_data_path 에 있는 데이터를 개별 슬라이드 단위로 분리한 시각적 이미지 정보(npy 형식으로 생성한 파일)가 저장되는 위치입니다.
            4. slide_metainfo_gen
                1. enable
                    1. 기본값 : **true**
                    2. 개별 슬라이드 단위로 분할된 이미지 파일의 색인 정보를 담고 있는 파일을 생성하는 switch 입니다.
                2. output_path
                    1. 기본값 : ${paths.data_root_dir}train_sampled_new.csv
                    2. 개별 슬라이드 단위로 분할된 이미지 파일의 색인 정보를 담고 있는 파일이 저장되는 위치입니다.
            5. preprocess
                1. crop
                    1. enable
                        1. 기본값 : **true**
                        2. 개별 슬라이드 단위로 분할된 이미지를 저장할 때 crop 기능을 적용하는 switch 입니다.
                    2. info_file_path
                        1. 기본값 : ${paths.data_root_dir}series_crop_bbox_loss_20260129_153709.npy **(예시)**
                        2. crop 처리에 사용될 ROI bounding box 정보가 담긴 파일의 위치이며, Stage 1 predict 단계 산출물(2.ii.b.a.c vessel_roi_predict.predict.result_npy_file_name 설정)입니다.
                2. resize
                    1. enable
                        1. 기본값 : false
                        2. 개별 슬라이드 단위로 분할된 이미지를 저장할 때 resize 기능을 적용하는 switch 입니다.
    2. 전처리된 개별 슬라이드 단위의 데이터를 바탕으로 **14개의 label에 대하여 이진 분류를 하도록 모델을 훈련**시킵니다.
        1. 실행 방법
            1. 기본 방법 : "python [./task/classifier_train.py](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/task/classifier_train.py)"
            2. 설정 override 를 위해 아래와 같이 experiment 설정을 적용할 수 있습니다.
                1. python [./task/classifier_train.py](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/task/classifier_train.py) experiment=[00_02_10_confirm_classification_setting](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/_configs/experiment/00_02_10_confirm_classification_setting.yaml)
        2. 주요 설정
            1. data ([./_configs/data/default.yaml](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/_configs/data/default.yaml) 파일 참고)
                1. common
                    1. label_file_name
                        1. 기본값 : train.csv
                        2. kaggle 에서 제공한 series 의 기본 정보를 담고 있는 파일의 이름을 설정하면 됩니다.
                    2. localizer_label_file_name
                        1. 기본값 : train_localizers.csv
                        2. kaggle 에서 제공한 positive(뇌동맥류 특징이 강한) 슬라이드 정보를 담고 있는 파일의 이름을 설정하면 됩니다.
                2. classifier
                    1. series_slide_fileset_path
                        1. 기본값 : ${paths.data_root_dir}raw_slide_data_cropped_loss_20260129_153709
                        2. kaggle 에서 제공한 DICOM series 데이터를 개별 슬라이드 단위로 분리한 시각적 이미지 정보(npy 형식으로 생성한 파일)가 존재하는 위치입니다. 1.i.b.c(dcm_to_npy.output_path) 과 동일한 대상을 가리킵니다.
                    2. series_slide_metainfo_file_name
                        1. 기본값 : train_sampled_new.csv
                        2. 개별 슬라이드 이미지 파일의 색인 정보를 저장한 파일 이름입니다. 1.i.b.d.b(dcm_to_npy.slide_metainfo_gen.output_path) 설정의 파일 이름과 같습니다.
                    3. roi_crop_info_file_name
                        1. 기본값 : '' or series_crop_bbox_loss_20260129_153709.npy **(예시)**
                        2. 예측된 ROI bounding box 정보가 저장된 파일의 이름을 지정합니다. Stage 1의 주요 산출물로 2.ii.b.a.c(vessel_roi_predict.predict.result_npy_file_name) 설정값에 npy 확장자(.npy)가 연결된 형식입니다. 3.i 과정을 생략한다면 crop 처리를 위해서는 반드시 지정해야하며 3.i 과정을 통해 series_slide_fileset_path 위치에 있는 데이터가 이미 crop 처리되었다면 빈 문자열로 설정되어야 합니다.
                    4. num_series_slide
                        1. 기본값 : 64
                        2. 뇌동맥류 negative 데이터에만 적용되는 학습 데이터에 포함되는 슬라이드 최대 개수 입니다. 참고한 **Harshit Sheoran**의 솔루션에는 negative 데이터(series 단위)의 경우 모든 슬라이드를 학습 데이터에 포함했지만 그럴 경우 데이터의 양이 많아 훈련 시간도 그만큼 오래 걸립니다. 해당 값 이하의 슬라이드 개수를 가진 series 의 경우 모든 슬라이드를 포함하고 그렇지 않은 series에 대해서는 해당 값만큼의 등간격 슬라이드만을 학습 대상에 포함시킵니다.
                    5. base_transform : train/valid 상관없이 모든 dataset 에 적용하는 이미지 변환입니다.
                    6. aug_transform : train dataset 에만 적용하는 이미지 데이터 증강 변환입니다.
            2. model.classifier ([./_configs/model/default.yaml](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/_configs/model/default.yaml) 파일 참고)
                1. backbone
                    1. model_name
                        1. 기본값 : resnet18
                        2. 분류 모델의 backbone을 구성하는 모델의 종류입니다.
            3. trainer.classifier ([./_configs/trainer/default.yaml](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/_configs/trainer/default.yaml) 파일 참고)
                1. save_path
                    1. 기본값 : ${paths.save_folder}
                    2. 훈련된 모델의 가중치가 저장되는 위치입니다.
                2. predict_save_path
                    1. 기본값 : ${result_dir}/predict/
                    2. 최상의 모델 가중치를 기반으로 validate dataset에 대한 예측값이 저장되는 위치입니다. 이는 이후 단계에서 positive 슬라이드 데이터를 증강하거나 soft target 데이터를 구성하는데 사용됩니다.
    3. 앞서 훈련된 분류 모델을 활용하여 임의로 설정한 특정 임계값을 기준으로 positive 슬라이드로 활용할 수 있는 데이터를 선별합니다.
        1. 실행 방법 : "python [./task/classify_predict_select.py](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/task/classify_predict_select.py)"
        2. 주요 설정 ("[./_configs/classify_predict_select.yaml](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/_configs/classify_predict_select.yaml)" 파일 참고)
            1. slide_metainfo_file_path
                1. 기본값 : ${paths.data_root_dir}train_sampled_new.csv
                2. 개별 슬라이드 이미지 파일의 색인 정보를 저장한 파일 이름입니다. 1.i.b.d.b(dcm_to_npy.slide_metainfo_gen.output_path) 설정과 동일합니다.
            2. fold_result_root_paths
                1. 기본값 : [
                    './_results/00_02_10_confirm_classification_setting/predict/20260216_192615',
                    './_results/00_02_10_confirm_classification_setting/predict/20260217_152413',
                    './_results/00_02_10_confirm_classification_setting/predict/20260218_110538',
                    './_results/00_02_10_confirm_classification_setting/predict/20260219_031045',
                    './_results/00_02_10_confirm_classification_setting/predict/20260219_224753',
                    ] **(예시)**
                2. 분류 모델을 통해 예측된 정보가 있는 위치(3.ii.b.c.b(trainer.classifier.predict_save_path) 설정 참고)를 설정합니다. 5 fold 데이터로 분류 모델을 훈련한 경우 위와 같이 index에 맞게 경로 목록을 구성합니다.
            3. use_soft_target
                1. 기본값 : **false**
                2. soft target 기반 정보를 데이터에 포함할지 결정하는 switch 입니다. 현 단계에서는 비활성화합니다.
            4. positive_threshold
                1. 기본값 : **1.6**
                2. positive/negative 를 구분하는 기준값입니다. 14개의 이진 분류 값은 0.0 ~ 1.0의 값을 갖는데 aneurysm_present(뇌동맥류 존재 여부)가 positive인 대부분의 경우 13개의 위치 라벨 중 최소 1개의 라벨에서도 positive 값을 갖습니다. 따라서 2개의 라벨에서 (평균적으로) 0.8을 초과하는 경우 positive 슬라이드로 구분됩니다. 이는 통계 분석 후 임의의 선택 및 결정에 따라 변경할 수 있습니다.
            5. positive_data_output_path
                1. 기본값 : ${paths.data_root_dir}train_sampled_positive.csv
                2. 증강 대상인 positive 데이터 색인 정보가 slide_metainfo_file_path의 파일과 동일한 형식으로 저장되는 위치 및 파일 이름입니다.
    4. 3.iii 단계에서 선별한 positive 슬라이드를 학습 데이터에 추가하여 **14개의 label에 대하여 이진 분류를 하도록 모델을 다시 훈련**시킵니다.
        1. 실행 방법
            1. 기본 방법 : "python [./task/classifier_train.py](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/task/classifier_train.py)"
            2. 설정 override 를 위해 아래와 같이 experiment 설정을 적용할 수 있습니다.
                1. python [./task/classifier_train.py](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/task/classifier_train.py) experiment=[00_03_00_expand_positive_label_data](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/_configs/experiment/00_03_00_expand_positive_label_data.yaml)
        2. 주요 설정 **(3.ii 단계에서 설정한 내용과 동일한 항목의 경우 생략하였습니다.)**
            1. data ([./_configs/data/default.yaml](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/_configs/data/default.yaml) 파일 참고)
                1. classifier
                    1. expand_metainfo_file_name
                        1. 기본값 : train_sampled_positive.csv
                        2. 증강 대상인 positive 슬라이드 데이터 정보를 저장한 파일을 지정합니다. 3.iii.b.e(classify_predict_select.positive_data_output_path) 설정의 파일 이름과 같습니다.
                    2. use_soft_target
                        1. 기본값 : false
                        2. expand_metainfo_file_name 에서 설정한 파일에 담긴 증강 슬라이드를 학습할 때 손실 측정에 soft target 을 사용할지 결정하는 switch 입니다. 현 단계에서는 비활성화 합니다.
    5. 3.iv 단계에서 훈련된 모델을 활용하여 positive 슬라이드를 대상으로한 지식 증류(Knowledge Distillation) 기법의 soft target 정보를 구성합니다.
        1. 실행 방법 : "python [./task/classify_predict_select.py](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/task/classify_predict_select.py)"
        2. 주요 설정 ("[./_configs/classify_predict_select.yaml](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/_configs/classify_predict_select.yaml)" 파일 참고)
            1. slide_metainfo_file_path
                1. 기본값 : ${paths.data_root_dir}train_sampled_new.csv
                2. 개별 슬라이드 이미지 파일의 색인 정보를 저장한 파일 이름입니다. 1.i.b.d.b(dcm_to_npy.slide_metainfo_gen.output_path) 설정과 동일합니다.
            2. fold_result_root_paths
                1. 기본값 : [
                    './_results/00_03_00_expand_positive_label_data/predict/20260220_182555',
                    './_results/00_03_00_expand_positive_label_data/predict/20260221_143544',
                    './_results/00_03_00_expand_positive_label_data/predict/20260222_043101',
                    './_results/00_03_00_expand_positive_label_data/predict/20260223_004947',
                    './_results/00_03_00_expand_positive_label_data/predict/20260223_201131',
                    ] **(예시)**
                2. 분류 모델을 통해 예측된 정보가 있는 위치(3.ii.b.c.b(trainer.classifier.predict_save_path) 설정 참고)를 설정합니다. 5 fold 데이터로 분류 모델을 훈련한 경우 위와 같이 index에 맞게 경로 목록을 구성합니다.
            3. use_soft_target
                1. 기본값 : **true**
                2. soft target 기반 정보를 생성하여 데이터에 반영합니다.
            4. positive_data_output_path
                1. 기본값 : ${paths.data_root_dir}train_sampled_soft_target.csv
                2. 증강 대상인 positive 데이터 색인 정보가 slide_metainfo_file_path의 파일과 동일한 형식으로 저장되는 위치 및 파일 이름입니다.
    6. 3.v 단계에서 선별한 positive 슬라이드를 학습 데이터에 추가하여 **14개의 label에 대하여 이진 분류를 하도록 모델을 다시 훈련**시킵니다. 현재 단계에서는 증강된 positive 슬라이드의 경우 soft target 을 바탕으로 손실값을 산정합니다.
        1. 실행 방법
            1. 기본 방법 : "python [./task/classifier_train.py](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/task/classifier_train.py)"
            2. 설정 override 를 위해 아래와 같이 experiment 설정을 적용할 수 있습니다.
                1. python [./task/classifier_train.py](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/task/classifier_train.py) experiment=[00_03_01_expand_soft_target_label_data](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/_configs/experiment/00_03_01_expand_soft_target_label_data.yaml)
        2. 주요 설정 **(3.2 단계에서 설정한 내용과 동일한 항목의 경우 생략하였습니다.)**
            1. data ([./_configs/data/default.yaml](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/_configs/data/default.yaml) 파일 참고)
                1. classifier
                    1. expand_metainfo_file_name
                        1. 기본값 : train_sampled_soft_target.csv
                        2. 증강 대상인 positive 슬라이드 데이터 정보를 저장한 파일을 지정합니다. 3.v.b.d(classify_predict_select.positive_data_output_path) 설정의 파일 이름과 같습니다.
                    2. use_soft_target
                        1. 기본값 : **true**
                        2. expand_metainfo_file_name 에서 설정한 파일에 담긴 증강 슬라이드를 학습할 때 손실 측정에 soft target 을 사용합니다.

4. 추론 : 실제 의학 영상 자료에 대한 모델 사용 사례
    1. kaggle 에서 제시하는 모델 평가 방법에 맞게 추론 과정을 구현하였습니다. 1건의 진단으로 획득한 DICOM 형식의 슬라이드 파일 집합을 하나의 폴더에 모아둔 경우 별다른 추가 작업없이 적용 가능합니다.
        1. 실행 방법 : "python [./task/for_submission.py](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/blob/main/task/for_submission.py)"
