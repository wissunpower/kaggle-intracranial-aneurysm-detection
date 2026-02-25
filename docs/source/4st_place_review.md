## RSNA2025 4st Place Review
#### [Kaggle Solution 링크](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution)
#### [Kaggle Train Source Code 링크](https://www.kaggle.com/datasets/harshitsheoran/rsna2025-training-code)
#### 개요
+ 데이터 전처리
  - 파일 형식daicom series -> npy
    - 결과물: **train_sampled** 폴더 및 **train_sampled.csv** 파일
    - 이미지 크기가 주류(Counter.most_common)인 데이터만 대상으로 설정
    - series 슬라이드 개별 단위로 저장
  - Fold 분할
    - 결과물: **train_fold.csv** 파일
    - fold 크기는 4

+ Segmentation(ROI) Crop
  - 데이터 전처리
    - 결과물: **segmentations_window1** 폴더 및 **segmentations_window1.csv** 파일
    - segmentations/*_cowseg.nii 파일 segment 정보 기준으로 0보다 큰 값에 대한 영역 산출
    - channel(혹은 depth) 차원의 크기를 특정 값(48)으로 조정하여 이미지 정보(numpy.ndarray)를 저장 (segment 정보가 없는 데이터도 포함, 4348 모두)
      - **train_windows1_test1/*.npy**
  - ROI 영역 추출 모델 훈련
    - segment 데이터(178개)와 이에 해당하는 series 데이터를 대상으로 ROI 영역(x1, x2, y1, y2)을 추출하도록 훈련
    - 대상 fold 는 0, 1
    - 48개의 등 간격 이미지를 3개씩 묶어 16개의 3channel 형식으로 모델에 입력
    - (batch_size * 16, 3, 128, 128) 로 (x1, x2, y1, y2) 를 예측하도록 훈련
  - 모든 case 에 대한 ROI 영역 추출(추론)
    - 결과물: **series_to_crop_try5smallplusv7.npy**
    - 훈련된 모델로 4348개의 모든 ROI를 추출(추론)

+ 분류
  - 대상 데이터
    - 동맥류 양성인 데이터의 경우 train_localizers.csv 에 포함된 단일 series 슬라이드만 포함, 음성인 데이터는 모든 슬라이드를 포함
    - current index 기준 (-2, 0, 2) channel 에 해당하는 단일 channel 이미지 3개를 묶음
    - 앞서 구성한 **series_to_crop_try5smallplusv7.npy**를 참조하여 ROI 영역에 magine 10%(양쪽 포함 총 20%) 를 추가한 부분이 학습 대상
  - 14개의 다중 이진 label 에 대한 분류 학습
  - 파이프라인 버전 특징(변경점)
    - try7
      - 모델 아키텍쳐는 **coat_lite_medium_384.in1k** 로 동일
      - v6
        - 훈련 완료 후 보완을 위한 positive label 데이터 추출
          - 결과물: **coat_lite_medium_384.in1k_v6/filt1.csv** 파일
          - positive 기준: aneurysm_present col 값이 1.6 초과한 경우
          - series 슬라이드 단위
      - v8
        - positive label 정보 참조 방식 개선
          - dataset.__getitem__에서 train_localizers.csv 에 없는 경우 train.csv 조회 시도
        - 훈련 완료 후 보완을 위한 positive label 데이터 추출
          - 결과물: **coat_lite_medium_384.in1k_v8/filt1.csv** 파일
          - positive 기준: aneurysm_present col 값이 2.0 초과한 경우
          - series 슬라이드 단위
      - v9
        - 학습 데이터에 추가 positive label 데이터 추가
          - positive 데이터 비율 증대
          - v6 기준
      - v10
        - 학습 데이터에 추가 positive label 데이터 추가
          - positive 데이터 비율 증대
          - v8 기준
      - v13
        - positive label 정보 참조 방식 개선
          - train_localizers.csv 에서 coordinates column 의 f 필드 값으로 positive series 슬라이드를 판정하도록 우선 시도
    - try8
      - 지식 증류(Knowledge Distillation) 기반 soft target 데이터 구성
        - 결과물: **distil_data1_coatv8v9v10v13.csv** 파일
        - try7 과정의 v8, v9, v10, v13 결과를 앙상블(균등 비율 0.25)하여 산정
      - 훈련 대상 데이터: 기본 데이터 + 지식 증류 기반 soft target 데이터
      - coat_lite_medium_384.in1k_v3
      - maxvit_tiny_tf_384.in1k_v1
      - maxvit_small_tf_384.in1k_v1

+ 추론
  - 앙상블 기반 추론
    - try7
      - coat-lite-medium-384-in1k-v13
    - try8
      - coat-lite-medium-384-in1k-v3
      - maxvit-tiny-tf-384-in1k-v1
      - maxvit-small-tf-384-in1k-v1
  - 입력 형식: series 개수 * 3 * 이미지 높이 * 이미지 너비
  - 결과 산정: (series 개수 * 14)에서 **series 축 기준으로 max값** 적용
