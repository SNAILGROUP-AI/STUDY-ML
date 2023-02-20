# Machine Learning

## 기계학습이란 무엇인가

### 정의
- 컴퓨터 프로그램이 주어진 데이터를 통해 데이터 처리 경험을 훈련함으로써 정보 처리 능력을 향상시키는 행위

### 목적
- 가지고 있는 데이터 X를 학습하여 생성한 함수에 넣으면, 그 결과로 어떠한 문제에 대한 예측치 Y를 반환함
- 기계학습은 우리가 찾고자 하는 함수를 찾아냄

### 알고리즘과 모델
- 알고리즘 : 어떠한 문제를 해결하기 위한 일련의 절차나 방법
- 모델 : 상관관계를 식으로 표현한 것으로서 알고리즘을 통해 도출된 설명변수와 반응변수의 관계를 나타낸 함수
   
---   
   
## 기계학습의 분류

### 사람의 감독 하에 훈련하는가?

#### 지도학습(Supervised Learning)

- 학습 시 설명변수에 대한 반응변수를 함께 제시함
- 일고리즘은 설명변수와 반응변수의 상관관계를 가장 잘 설명할 수 있는 모델을 찾음
- 알고리즘은 모델을 사용하여 새로운 설명변수에 대하여 예측을 수행함
- 예시

| 학습방식 | 분석 종류 | 예시 |
|---|---|---|
| 지도학습 | 분류분석 | 결정트리(Decision Tree) |
| | | 서포트 벡터 머신(Support Vector Machine) |
| | | k-최근접이웃(K-Nearest Neightbor: KNN) |
| | | 로지스틱 회귀(Logistic Regression) |
| | 회귀분석 | 결정트리(Decision Tree) |
| | | 선형 회귀(Linear Regression) |
| | | 확률적 경사 하강 회귀(Stochastic gradient descent Regression; SGD) |
| 비지도학습 | 군집분석 | K-Means |
| | | 계층적 군집 분석(Hierarchical Cluster Analysis; HCA) |
| | | DBSCAN | 
| | 시각화와 차원 축소 | 주성분 분석(Principal Component Analysis; PCA) |
| | | 커널 주성분 분석(Kernel Principal Component Analysis) |
| | | 지역적 선형 임베딩(Locally-Linear Embedding; LLE) |
| | | t-SNE(t-distributed Stochastic Neighbor Embedding) |
| | 이상치 탐지 | 가우스 분포 |
| | 연관규칙 | Apriori |
| | | Eclat |


#### 비지도학습(Unsupervised Learning)

- 학습 시 설명변수에 대한 반응변수를 제시하지 않음
- 알고리즘은 설명변수의 특징만을 활용하여 목표한 결과를 산출함
- 예시

#### 준지도학습 (Semi-supervised Learning)

#### 강화학습 (Reinforcement Learning)

#### 전이학습 (Transfer Learning)


### 실시간으로 점진적인 학습을 하는가?

#### 온라인 학습

- 데이터를 개별적으로, 또는 소그룹(mini batch)으로 묶어서 순차적으로 제공하며 점진적으로 훈련시킴    
- ![온라인 학습](https://tensorflowkorea.files.wordpress.com/2018/05/e18489e185b3e1848fe185b3e18485e185b5e186abe18489e185a3e186ba-2018-05-24-e1848be185a9e18492e185ae-6-23-50.png?w=768)   


#### 오프라인 학습

- 배치학습이라고도 함
- 시스템이 점진적으로 학습할 수 없는 경우
- 먼저 시스템을 훈련시킨 후, 제품에 적용하면 더 이상의 업데이트 없이 실행됨
- 모든 데이터를 사용하여 학습함
- 많은 시간이 소요되고 많은 리소스가 동원되므로 오프라인으로 수행함

    
### 사례 기반 vs 모델 기반

#### 인스턴스 기반 학습

#### 모델 기반 학습