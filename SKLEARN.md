# [Scikit-Learn Library](https://scikit-learn.org/stable/#)  

---

## 정의  

### 정의

- 머신러닝 기술을 통일된 인터페이스로써 활용할 수 있도록 정리한 라이브러리  

- 머신러닝 알고리즘, 머신러닝 개발을 위한 프레임워크 및 API 제공  

### API 사용 방법

- 적절한 estimator 클래스를 임포트

- 클래스의 하이퍼파라미터를 적절한 값으로 설정하여 인스턴스 생성

- 데이터를 피쳐(속성)와 타깃(정답)으로 배치
    - 행(row) : 자료, 인스턴스, 튜플 등  
    - 열(column) : 속성, 피쳐, 필드, 어트리뷰트 등

- 인스턴스의 메소드 `fit()`을 통해 인스턴스를 학습용 데이터로 훈련시킴

- 인스턴스의 메소드 `predict()`을 통해 훈련된 인스턴스에 테스트용 데이터를 적용함

### [권장 사항](https://scikit-learn.org/stable/modules/classes.html)

![이미지](https://scikit-learn.org/stable/_static/ml_map.png)

---

## 주요 모듈

<details><summary><h3>주요 모듈</h3></summary>

| 모듈 | 설명 | 예시 |
|------|------|------|
| sklearn.tree | 결정 트리 알고리즘 제공 |
| sklearn.neighbors | 최근접 이웃(K-NN) 알고리즘 제공 |
| sklearn.svm | 서포트 벡터 머신 알고리즘 제공 |
| sklearn.naive_bayes | 나이브 베이즈 알고리즘 제공 | 가우시안 NB, 다항 분포 NB 등 |
| sklearn.cluster | 클러스터링 알고리즘 제공 | K-Means, 계층형 클러스터링, DBSCAN 등 |
| sklearn.linear_model | 회귀분석 관련 알고리즘 제공 | 선형 회귀, 확률적 경사하강 회귀(SGD), 릿지(Ridge), 라쏘(Lasso), 로지스틱 회귀 등 |
| sklearn.decomposition | 차원 축소 관련 알고리즘 제공 | PCA, NMF, Truncated SVD 등 |
| sklearn.ensemble | 앙상블 알고리즘 제공 | Random Forest, AdaBoost, GradientBoost 등 |
| sklearn.preprocessing |데이터 전처리 기능 제공 | 인코딩, 스케일링 등 |
| sklearn.feature_selection | 특성(feature)를 선택할 수 있는 기능 제공 | 
| sklearn.feature_extraction | 특성(feature)을 추출할 수 있는 기능 제공 |
| sklearn.pipeline | 특성 처리, 학습, 예측을 묶어서 실행할 수 있는 기능 제공 |
| sklearn.model_selection | 교차 검증, 최적 하이퍼파라미터 추출 API 제공 | GridSearch 등 |
| sklearn.metrics | 성능 측정 방법 제공 | Accuracy, Precision, Recall, ROC-AUC, RMSE 등 |
| sklearn.datasets | 내장 예제 세트 제공 |

</details>

<details><summary><h3>sklearn.datasets 내장 데이터 셋</h3></summary>

| 데이터 로드 함수 | 데이터 | 참고 |
|------|------|------|
| load_boston | 보스턴 집값 | 내장 데이터  |
| load_diabetes | 당뇨병 |  |
| load_linnerud | linnerud |  |
| load_iris | 붓꽃 |  |
| load_digits | 필기 숫자(digit) 이미지 |  |
| load_wine | 포도주(wine) 등급 |  |
| load_breast_cancer | 유방암 진단 |  |
| fetch_california_housing | 캘리포니아 집값 | 인터넷 다운로드 |
| fetch_covtype | 토지조사 |  |
| fetch_20newsgroups | 뉴스 그룹 텍스트 |  |
| fetch_olivetti_faces | 얼굴 이미지 |  |
| fetch_lfw_people | 유명인 얼굴 |  |
| fetch_lfw_pairs | 유명인 얼굴 |  |
| fetch_rcv1 | 로이터 뉴스 말뭉치 |  |
| fetch_kddcup99 | Kddcup 99 Tcp dump |  |
| make_regression | 회귀분석용 | 가상 데이터 |
| make_classification | 분류용 |  |
| make_blobs | 클러스터링용 |  |

</details>

<details><summary><h3>sklearn.datasets 내장 데이터 형식</h3></summary>

| 이름 | 설명 |
|------|------|
| DESCR | 자료에 대한 설명 |
| data | 설명 변수 |
| target | 반응 변수 |
| feature_names | 설명 변수 이름 리스트 |
| target_names | 반응 변수 이름 리스트 |

</details>