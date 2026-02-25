## RSNA2025 1st Place Review
#### [Kaggle Solution 링크](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution)
#### [GitHub Train Source Code 링크](https://github.com/uchiyama33/rsna2025_1st_place)
#### 개요
+ 2 Stage(단계) 구조
+ 1 Stage
  - ROI 및 Semantic Segmentation 추출
  - Trainer Class 계층
    - nnUNetTrainerSkeletonRecall(nnUNetTrainer)
      - nnUNetTrainerSkeletonRecall_onlyMirror01
        - **nnUNetTrainerSkeletonRecall_more_DAv3 (1-1)**
          - Model 2: Fine Segmentation (Balanced)
          - RSNA2025Trainer_moreDAv6_SkeletonRecall
            - RSNA2025Trainer_moreDAv6_SkeletonRecallTverskyBeta07
              - **RSNA2025Trainer_moreDAv6_SkeletonRecallW3TverskyBeta07 (1-2)**
          - RSNA2025Trainer_moreDAv6_1_SkeletonRecall
            - **RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07 (1-3)**
    - nnUNetTrainer_onlyMirror01(nnUNetTrainer)
      - RSNA2025Trainer_moreDAv3
        - **RSNA2025Trainer_moreDAv7 (3)**
          - Model 1: Coarse Vessel Localization
  - Sementic Segment Backbone
    - dynamic_network_architectures.architectures.unet.ResidualEncoderUNet
    - input channels : 1
    - output channels : 14 (background + 13개의 혈관 class)
  - 입력: 178 개의 Semantic Segmentation 데이터
    - 전처리
      - RAS orientation 정규화
      - 값 정규화((값 - min) / max)
      - affine 변환 적용
    - forward
      - 입력 데이터
        - 공통적으로 random crop(patch_size 기반) 적용
        - 입력 값
          - data: (1, 1, 64, 128, 128), (patch_size: 64, 128, 128)
        - Label 값
          - -1 값을 0으로 변경
          - target: background 를 포함한 13개의 혈관 segment
            - downsampling 이 적용된 tensor list
          - skel: 혈관 union segment(one-hot이 아닌 label값 기반)
            - skimage.morphology 의 skeletonize, dilation 변환을 적용
      - 출력 값
        - skips output, 5회 downsampling
      - 손실 함수
        - downsampling 단계 중 해상도가 가장 높은 layer 에 대해서만 산정
        - dice
        - skeleton recall
        - cross entropy
  - Semantic Segmentation 추론
    - roi 및 seg shape 변경 기준으로 서술
      - 전치(transpose) 설정 적용
      - 0값인 가장자리 영역을 crop
      - spacing scale 적용
    - patch_size 보다 작을 경우 padding 처리
    - 원본 데이터를 patch_size 만큼 짤라 순회하여 voxel feature 획득
      - gaussian 기준 정규화
    - 일정한 threshold(임계값) 기준으로 bbox 추출 : foreground 성격
      - 해당 bbox 가 존재할 경우 해당 영역만 crop
    - 모든 extra model 에 대해 foreground box를 계속 override 한다.
+ 2 Stage
  - ROI 및 Semantic Segmentation 기반 분류
  - backbone
    - **RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07 (1-3)**
  - roi 및 vessel seg 를 input size(96, 192, 192 혹은 64, 128, 128) 에 최대한 맞추어 scale 하여 pad 후 crop
  - roi를 backbone 에 통과시킨 후 가장 큰 decode features 와 작은 encode features 획득
  - vessel seg 가 decode features 보다 depth, height, width 가 2배 큰데, 이를 줄여 맞춤
  - 각각의 vessel seg(13개)에 대한 vessel region masked pooling 단계
    - decode features mask recall 을 산정 -> (1)
    - encode features 에 global average pooling 을 적용 -> (2)
    - (1)과 (2)를 Location-aware-transformer 에 입력하여 나온 결과를 13개의 다중 라벨과 비교
  - vessel seg union에 대한 vessel region masked pooling 단계
    - decode features mask recall 을 구하여 이를 ap 라벨과 비교
#### 데이터
+ Dataset 출력
  - ROI 이미지
  - 세그먼트 마스크 라벨
  - 분류 라벨(14개)
+ Model forward 입력
  - ROI 이미지
  - 세그먼트 마스크 라벨
  - 세그먼트 union 마스크
