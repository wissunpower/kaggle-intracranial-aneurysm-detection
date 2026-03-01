# Intracranial Aneurysm Detection
![main screenshot](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/raw/main/docs/source/_static/img/kaggle_competitions_overview.png)

--------------------------------------------------------------------

[북미영상의학회](https://www.rsna.org/)에서 Kaggle을 통해 주최한 뇌동맥류 탐지 경연에 참가하여 작업한 내용을 담고 있습니다.

#### [Kaggle Competition 링크](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)

뇌동맥 일부가 약해져서 해당 부분이 풍선이나 꽈리처럼 부풀어 오르고 최악의 경우 파열되어 지주막하 뇌출혈과 같은 치명적 질환을 유발하는 것을 **뇌동맥류**라고 합니다.
(질환에 대한 자세한 정보 [link](https://health.kdca.go.kr/healthinfo/biz/health/gnrlzHealthInfo/gnrlzHealthInfo/gnrlzHealthInfoView.do?cntnts_sn=5963))

CT, MRI와 같은 의학 영상자료를 통해 해당 질환의 발현 및 그 가능성을 예측하는 것이 목적으로 결과는 뇌동맥류 존재 여부를 포함한 동맥류 위치 해당 여부를 다중 이진 분류 형식으로 표현됩니다.
모델의 전체적인 성능은 총 14개의 label 각각에 대해 [AUC ROC 점수](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=ko)를 계산하는 방식으로 평가됩니다.

## 데이터
+ **train.csv** 파일은 개별 의학영상 자료 series에 대한 기본 학습 label 정보를 담고 있습니다. 자료 유형(CTA, MRA, MRI와 같은)과 환자의 간단한 인적사항 및 동맥류 증상이 나타난 세부 혈관 위치 정보 등을 포함합니다.

+ **train_localizers.csv** 파일은 동맥류 위치에 대한 보다 자세한 정보를 제공합니다. series 슬라이드 묶음 중에서 positive 성격이 가장 강한 슬라이드의 instance UID 와 동맥류 중심 부근 좌표 정보를 확인할 수 있습니다.

+ **series/** 학습에 사용해야할 DICOM series 데이터 집합으로, 각각의 폴더는 1개의 series 를 나타내며, 파일 구조는 "series/{SeriesInstanceUID}/{SOPInstanceUID}.dcm"와 같습니다. 보통은 1개의 dcm 파일이 하나의 슬라이드 정보를 담고 있지만, 모든 슬라이드 정보를 오직 1개의 dcm 파일이 전부 가지고 있는 경우도 있습니다.

+ **segmentations/** series/ 에 있는 4348건의 DICOM seires 중에 일부인 178건에 대한 혈관 분할 정보가 있습니다.(NifTI 파일 형식)

혈관 세부 위치에 대한 label 구분은 아래와 같습니다.

| 레이블 Index   | Label   | 레이블   |
| ---------- | ------- | ------- |
| 0 | Left Infraclinoid Internal Carotid Artery | 좌측 하돌기 내측 경동맥 |
| 1 | Right Infraclinoid Internal Carotid Artery | 우측 하돌기 내측 경동맥 |
| 2 | Left Supraclinoid Internal Carotid Artery | 좌측 상극돌기 내경동맥 |
| 3 | Right Supraclinoid Internal Carotid Artery | 우측 상극돌기 내경동맥 |
| 4 | Left Middle Cerebral Artery | 좌측 중뇌동맥 |
| 5 | Right Middle Cerebral Artery | 우측 중뇌동맥 |
| 6 | Anterior Communicating Artery | 전방 교통 동맥 |
| 7 | Left Anterior Cerebral Artery | 좌측 전대뇌동맥 |
| 8 | Right Anterior Cerebral Artery | 우측 전대뇌동맥 |
| 9 | Left Posterior Communicating Artery | 좌측 후방 교통 동맥 |
| 10 | Right Posterior Communicating Artery | 우측 후방 교통 동맥 |
| 11 | Basilar Tip | 기저부 끝 |
| 12 | Other Posterior Circulation | 기타 후방 순환 |
| 13 | Aneurysm Present | 뇌동맥류 존재 여부 |

## 활동 요약
대회가 진행되는 동안에는 4348개의 series 인스턴스 dcm 데이터를 32 x 416 x 416 크기의 npy 데이터로 변환하여 이를 입력으로 받아 14개의 이진 분류 label을 생성하는 방식을 적용하였습니다.
3D 형식에 가까운 tensor 데이터를 기반으로 분류를 하는 단순 end to end 구조로 최종 공식 AUC ROC 점수는 0.54539에 도달하였습니다.(순위는 939위, https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/leaderboard) 사실상 무작위 추측 모델 수준에 머물렀습니다.

머신러닝에 대한 역량을 증진하기 위해 대회 종료 후 상위 입상자의 솔루션을 분석한 후 주요 구성요소를 필자의 프로젝트에 이식하여 성능을 끌어올리는 작업을 시도하였습니다.

처음에는 1위에 입상한 **tomoon33**(https://www.kaggle.com/tomoon33)의 솔루션을 살펴보았습니다.
#### RSNA2025 1st Place Review
##### [Kaggle Solution 링크](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution)
##### [GitHub Train Source Code 링크](https://github.com/uchiyama33/rsna2025_1st_place)
+ 주요 특징
  - semantic segmentation 방식으로 혈관에 대한 세부 segment와 ROI 를 추출한 다음 이를 바탕으로 13개의 혈관 세부 위치와 뇌동맥류 존재 여부를 분류하는 2개의 stage 로 구성되어 있습니다.
  - 첫번째 stage : 혈관 세부 segment 와 ROI를 추출하기 위한 semantic segmentation 단계
    - [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) 라이브러리 기반
    - nnUNetv2_plan_and_preprocess 와 같은 기능을 통해 데이터 전처리 및 학습을 위한 hyper-parameters 값 집합 획득
    - 의학 시각 자료에서의 혈관이나 위성 사진에서의 도로와 같은 가느다란 형태의 개체 segment를 탐색하는데 도움이 되는 [skeleton recall 기반 손실 함수](https://github.com/MIC-DKFZ/Skeleton-Recall) 적용
  - 두번째 stage : 첫번째 stage 결과물을 바탕으로 13개의 혈관 세부 위치와 뇌동맥류 존재 여부에 대한 classification 단계
    - 첫번째 stage 의 결과로 얻은 세부 위치 혈관 13종 각각의 segment로 0 ~ 12 의 label을 분류하며, 마지막 13인 뇌동맥류 존재 여부는 13개의 모든 union segment로 분류합니다.
+ 한계
  - 필자의 개발 환경 한계로 인해 nnUNet을 온전히 사용할 수 없었고, 첫번째 semantic segmentation stage 를 완료할 수 없었습니다.
    - 필자의 개발 환경
    ![gpu_info](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/raw/main/docs/source/_static/img/nvidia_smi_result.png)
  - [nnUNet 사용을 위한 하드웨어 권장 사양](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)
  - nnUNet preprocess 로 얻은 하이퍼파라미터에서 batch_size나 patch_size를 낮추어야만 겨우 학습을 진행할 수 있었으며,
  - 그래도 일부 데이터는 학습과정에서 exception이 발생하였고, 결정적으로 학습 완료 후 혈관 segment 및 ROI 를 추출하기 위한 inference 를 진행할 수 없었습니다.
+ 결실
  - 머신러닝 개발에 도움이 되는 Hydra, wandb 와 같은 utility 적용 사례를 참고할 수 있었습니다.
    - [Hydra for config composition](https://hydra.cc/docs/intro/) : 복잡한 설정 정보를 관리하는데 편의를 제공합니다.
    - [Weights & Biases](https://wandb.ai/site/ko/) : ML 결과를 관리하고 시각화하는데 도움을 줍니다.

제한적인 개발 환경에서도 다룰 수 있는 솔루션을 찾다가 4위에 입상한 **Harshit Sheoran**(https://www.kaggle.com/harshitsheoran)의 솔루션을 확인할 수 있었습니다.
#### RSNA2025 4st Place Review
##### [Kaggle Solution 링크](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution)
##### [Kaggle Train Source Code 링크](https://www.kaggle.com/datasets/harshitsheoran/rsna2025-training-code)
+ 주요 특징
  - object detection 방식으로 혈관의 모든 부위가 포함된 bounding box를 추출하고, 이를 기반으로 crop된 원본 이미지로 13개의 혈관 세부 위치와 뇌동맥류 존재 여부를 분류하는 2개의 stage 로 구성되어 있습니다.
  - 첫번째 stage : 혈관의 모든 부위를 포함하는 최소 영역을 추출하기 위한 bounding box 회귀 단계
    - segmentations/ 에 포함된 178건의 데이터로 모델 학습
    - 학습된 모델로 series/ 에 있는 4348건의 모든 데이터에 대한 ROI(bounding box) 추출
  - 두번째 stage : 혈관 부위로 최소화된 이미지를 기반으로 13개의 혈관 세부 위치와 뇌동맥류 존재 여부에 대한 classification 단계
    - 14개의 다중 이진 label 에 대한 분류 학습
    - 학습 데이터
      - 하나의 series 에 포함된 모든 슬라이드를 묶어 3D 형태의 tensor를 학습 데이터 단위로 사용하는 것이 아니라 개별 슬라이드 단위로 학습 데이터에 적용합니다. (총 데이터 수는 4,348이 아닌 1,028,811가 됩니다.)
      - 모델 탐지의 주요 목적인 **Aneurysm Present** label 이 1(positive)인 series의 경우 모든 슬라이드가 학습 대상이 되는 것은 아니며 train_localizers.csv 에 포함된 positive 성격이 가장 강한 슬라이드만 학습 대상으로 포함됩니다.
        - 결국 기본적인 학습 데이터의 수는 약 2,200개의 positive 슬라이드와 약 545,000개의 negative 슬라이드를 합한 약 547,200개가 됩니다.
      - 학습 단위 슬라이드는 하나 건너 앞의 슬라이드와 하나 건너 뒤의 슬라이드와 묶여 총 3개의 채널로 구성됩니다.
        - 크기가 284 x 240 인 15번째 슬라이드가 학습 대상이라면, 13, 15, 17번째 슬라이드가 묶여 3 x 284 x 240 의 크기를 갖는 데이터가 구성됩니다.
      - 학습 단위 슬라이드를 첫번째 stage 에서 추출된 ROI 정보를 바탕으로 crop 처리합니다.
    - 분류 학습은 1번으로 끝나지 않으며, 지식 증류(Knowledge Distillation) 기법을 통해 상대적으로 부족한 positive 슬라이드 데이터를 증강하면서 여러번 반복됩니다.

위와 같은 상위 입상자 솔루션의 분석 내용을 바탕으로 아래에 기술된 **인용 및 개선 사항**을 진행한 결과 모델의 성능을 향상시킬 수 있었습니다.

## 인용 및 개선 사항
+ ML 기능 관련
  - Pipeline 상의 변경 사항
    | 구 분 | Before | After |
    | :----------: | :-------: | :-------: |
    | 요 약 | 분류 단계만 있는 단순 End to End 구조 | ROI bounding box 추출 후 분류를 시도하는 2 Stage 구조 |
    | First Stage | - | 혈관 ROI에 대한 bounding box를 구분하기 위해 회귀(regresses) 학습 진행 |
    | First Stage 입력 데이터 | - | segmentations/ 에 있는 {SeriesInstanceUID}.nii 파일과 {SeriesInstanceUID}_cowseg.nii 파일 |
    | First Stage 출력 predict | - | bounding box를 가르키는 x1, x2, y1, y2 (0.0 ~ 1.0 범위로 정규화된 값) |
    | Second Stage | 뇌동맥류 존재 여부 및 동맥류 위치에 대한 14개 label의 다중 이진 분류 | 뇌동맥류 존재 여부 및 동맥류 위치에 대한 14개 label의 다중 이진 분류 |
    | Second Stage 입력 데이터 | 하나의 series 에 포함된 모든 슬라이드를 묶어 32 x 416 x 416 크기로 보간한 volume 형식 | 모든 series 의 모든 슬라이드를 개별로 구분하여 앞, 뒤로 인접한 슬라이드를 가져와 3 x 352 x 352 크기로 보간한 image 형식 |
  - label 값(positive or negative)에 따라 학습해야 할 슬라이드 선별
  - 지식 증류(Knowledge Distillation) 기법 사용
    - negative label 데이터에 비해 상대적으로 표본이 부족한 positive label 데이터 증강
    - 서로 다른 Fold 모델의 예측값을 바탕으로 soft target 학습 데이터를 구성하여 간접적인 앙상블 효과 반영
  - 평가 지표의 일관성을 높이기 위해 모델 가중치에 EMA 기법 적용
    - [revision d946d98](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/commit/d946d982141853385a72a260c0deaf96ba03793d)
+ 개발 편의 관련
  - 보다 편리한 hyper parameter 형상 관리를 위해 [Hydra](https://hydra.cc/docs/intro/) 적용
    - [revision 3b97940](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/commit/3b9794030f3503eb370ef5011c200d59fe3e9d8b)
  - ML 결과 분석에 편의를 제공하는 [wandb](https://wandb.ai/site/ko/) 연동
    - [revision 55cd8c5](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/commit/55cd8c549100ae632c229ea28f12c8498114716a)

## Requirements
본 repository의 python 스크립트를 실행하기 위해 필요한 주요 library 목록입니다.
특별히 명시된 항목을 제외하고는 유사한 Version 을 적용하시면 됩니다.
+ python  3.12.3
+ numpy  1.26.4
+ pandas  2.1.4
+ tqdm  4.67.1
+ rootutils  1.0.7
+ omegaconf  2.3.0
+ hydra  1.3.2
+ pydicom  3.0.1
+ nibabel  5.3.2
+ torch  2.9.0
+ scikit-base  0.7.8
+ scikit-learn  1.4.2
+ transformers  4.57.1
+ timm  1.0.20
  - 해당 버전보다 낮을 경우 최신 vision transformer 계열의 모델을 사용하는데 문제가 발생합니다.
+ wandb  0.23.0
+ albumentations 2.0.8
+ matplotlib  3.10.6

## 주요 Script 안내
대부분의 작업 단계는 ./task 폴더와 ./dataset/preprocess 폴더에 있는 python 스크립트를 실행하는 것으로 구성되어 있습니다. 또한 각각의 스크립트는 동일한 이름의 hydra 설정 파일(./_configs 폴더에 존재)로부터 실행하는데 필요한 정보를 입력 받으며, 크게 아래와 같이 구분됩니다.

(순서에 기반한 보다 구체적인 실행 안내는 [이곳](https://github.com/wissunpower/kaggle-intracranial-aneurysm-detection/tree/main/task)에서 확인하실 수 있습니다.)

+ 데이터 전처리
  - ./dataset/preprocess/dcm_to_npy.py
    - 4348개의 series 데이터를 개별 슬라이드 단위로 분리하여 시각적 이미지 정보는 npy 형식으로 저장하고, 모든 슬라이드 색인 정보를 각각의 row 단위로 매핑한 csv 파일을 생성합니다.
  - ./task/vessel_seg_to_roi_bbox_preprocess.py
    - segmentations/ 안에 혈관 segment 정보가 있는 series 데이터에 한하여 혈관 ROI label(bounding box 형식)을 생성합니다. 또한 모든 4348건의 series 데이터에 대한 슬라이드 이미지 정보를 ROI 영역 추출의 입력 데이터로 사용할 수 있게 전처리하여 저장합니다.
+ 혈관 ROI 추출 모델 훈련 : Stage 1
  - ./task/vessel_roi_bbox_train.py
    - 전처리된 segmentations/ 안의 혈관 segment 데이터로 **혈관 ROI bounding box를 예측하도록 모델을 훈련**시킵니다.
  - ./dataset/preprocess/vessel_roi_predict.py
    - './task/vessel_roi_bbox_train.py' 에서 훈련된 모델을 사용하여 4348건의 모든 series 데이터에 대한 혈관 ROI bounding box 추출을 진행합니다.
+ 뇌동맥류 질환 존재 여부 및 위치에 대한 다중 이진 분류 훈련 : Stage 2
  - ./task/classifier_train.py
    - './dataset/preprocess/dcm_to_npy.py' 에서 전처리된 슬라이드 단위의 데이터에 './dataset/preprocess/vessel_roi_predict.py' 과정에서 얻은 ROI 정보를 바탕으로 crop 한 후 **14개의 label에 대하여 이진 분류를 하도록 모델을 훈련**시킵니다.
  - ./task/classify_predict_select.py
    - './task/classifier_train.py' 에서 훈련된 모델을 활용하여 임의로 설정한 특정 임계값을 기준으로 positive 슬라이드로 활용할 수 있는 데이터를 선별합니다. 또한 지식 증류(Knowledge Distillation) 기법을 위한 soft target 정보가 포함된 슬라이드 데이터를 생성할 수 도 있습니다.
+ 추론 : 실제 의학 영상 자료에 대한 모델 사용 사례
  - ./task/for_submission.py
    - kaggle 에서 제시하는 모델 평가 방법에 맞게 추론 과정을 구현하였습니다. 1건의 진단으로 획득한 DICOM 형식의 슬라이드 파일 집합을 하나의 폴더에 모아둔 경우 별다른 추가 작업없이 적용 가능합니다.

## 부록 : debugging configuration launch.json for vscode
```
{
    "version": "0.2.0",
    "configurations": [
    
        {
            "name": "train with experiment",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            // "justMyCode": false,
            "console": "integratedTerminal",
            "args": [
                "experiment=00_03_01_expand_soft_target_label_data",
                "data.common.current_fold=3",
            ]
        },

    ]
}
```
