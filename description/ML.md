<img src="https://capsule-render.vercel.app/api?type=cylinder&color=F5ECCE&height=100&section=header&text=What?%20Machine%20Learning&fontSize=40" width=100%/>

</br>

## 👨‍🔧 머신러닝이란 무엇인가

<details><summary><h3>정의</h3></summary>

- **정의**
    - 컴퓨터 프로그램이 주어진 데이터를 통해 데이터 처리 경험을 훈련함으로써 정보 처리 능력을 향상시키는 행위
    
    - 설명변수 X와 반응변수 Y에 대하여, 두 변수 간 상관관계를 서술한 수식을 찾는 능력을 향상시키는 행위
    
    - 알고리즘을 통해 모델을 설계하는 행위
        - **알고리즘** : 어떠한 문제를 해결하기 위한 일련의 절차나 방법
        - **모델** : 상관관계를 식으로 표현한 것으로서 알고리즘을 통해 도출된 설명변수와 반응변수의 관계를 나타낸 함수

- **인공지능, 딥러닝과의 차이**
    - **인공지능(Artificial Intelligence; AI)** : 사람처럼 학습하고 추론할 수 있는 시스템을 만드는 기술
    - **머신러닝(Machine Learning; ML)** : 규칙을 프로그래밍하지 않아도 주어진 데이터에서 자동으로 규칙을 발견하는 시스템을 만드는 기술
    - **딥러닝(Deep Learning; DL)** : 인공 신경망을 기반으로 하는 머신러닝 기술

- **장점**
    - **전통적인 방식으로는 해결할 수 없는 복잡한 문제를 쉽게 풀 수 있음**
        - 전통적인 방식의 경우 개발자가 직접 규칙을 설계하고 파라미터를 조정해야 함
        - 머신러닝의 경우 개발자가 적절한 알고리즘을 채택하고 적절한 파라미터를 설정하면 모델이 스스로 규칙을 찾아냄
        - 따라서 전통적인 방식에 비해 머신러닝으로 문제를 풀 경우 개발자가 직접 처리해야 하는 업무가 줄어듦

    - **유동적인 환경에 대처할 수 있음**
        - 대용량 데이터로부터 스스로 규칙을 찾아내어 복잡한 문제에 대응함
        - 따라서 학습되지 않은 데이터에 대해서도 적용할 수 있음

- **빅데이터와의 관계**
    - 빅데이터의 개념
        - **빅데이터(big data)** : 기존의 데이터베이스로는 수집, 처리, 저장, 분석을 수행하기 어려울 만큼 방대한 양의 데이터
        - **빅데이터 시스템(big data system)** : 빅데이터를 다루기 위한 시스템
        - **빅데이터 엔지니어링(big data engineering)** : 빅데이터를 다루는 방법
    
    - 빅데이터와 머신러닝의 관계
        - 본래 빅데이터의 개념은 데이터베이스에서 기원하여 머신러닝과는 별개로 발전해왔음
        - 학습 가능한 데이터의 양이 머신러닝 모델의 성능을 좌우하게 되면서, 오늘날 머신러닝 분야에서 유의미해짐

- **이슈**
    - **데이터 문제**
    
        - 충분하지 않은 양의 데이터를 통한 학습
        - 대표성이 없는 데이터를 통한 학습
        - 품질이 낮은 데이터를 통한 학습
        - 반응변수와의 연관성이 낮은 설명변수를 통한 학습

    - **과적합 문제**

        ![과대적합과 과소적합](https://tensorflowkorea.files.wordpress.com/2017/06/fig2-01.png?w=640)

        - 모델이 훈련 시 제공되는 데이터에 과대 혹은 과소 적합되는 경우
        - 즉, 모델이 새로운 사례에 대하여 일반화되지 않는 경우
    
</details>

<details><summary><h3>WorkFlow</h3></summary>

![머신러닝워크플로우](https://content.altexsoft.com/media/2017/04/Screenshot_3.png)

- **Collect data** : 유용한 데이터를 최대한 많이 확보하고 하나의 데이터 세트로 통합함

- **Prepare data** : 결측값, 이상값, 기타 데이터 문제를 적절하게 처리하여 사용 가능한 상태로 준비함

- **Split data** :  데이터 세트를 학습용 세트와 평가용 세트로 분리함

- **Train a model** : 학습용 데이터 세트의 일부를 통해 모델이 데이터 내 패턴을 찾도록 훈련함

- **Validate a model** : 학습용 데이터 세트의 나머지를 통해 모델이 데이터 내 패턴을 잘 찾아냈는지 확인함

- **Test a model** : 평가용 데이터 세트를 통해 모델의 성능을 파악함

- **Deploy a model** : 모델을 의사결정 시스템에 탑재함

- **Iterate** : 새로운 데이터를 확보하고 모델에 적용하여 모델을 점진적으로 개선해나감

</details>

<details><summary><h3>주요 라이브러리</h3></summary>

| 용도 | 라이브러리명 |
|---|---|
| 머신러닝 | Scikit-Learn |
| 딥러닝 | Tensorflow, Keras, Pytorch |
| 수리통계 | NumPy, SciPy |
| 데이터 핸들링 | Pandas |
| 데이터 시각화 | Matplotlib, Seaborn, Plotly |

</details>

---

## 📚 머신러닝의 분류

<details><summary><h3>구분</h3></summary>

![머신러닝 구분](https://github.com/trekhleb/homemade-machine-learning/blob/master/images/machine-learning-map.png?raw=true)

</details>

<details><summary><h3>사람의 감독 하에 훈련하는가</h3></summary>

- **지도 학습(Supervised Learning)**
    - **정의 : 훈련 단계에서 설명변수의 조합에 대응하는 반응변수를 함께 제시하는 학습 방법**
        - 일고리즘은 설명변수와 반응변수의 상관관계를 가장 잘 설명할 수 있는 모델을 찾음
        - 알고리즘은 모델을 사용하여 새로운 설명변수에 대하여 예측을 수행함
    
    - **주요 알고리즘**

        | 분석 종류 | 알고리즘 |
        |---|---|
        | 분류분석 | 결정트리(Decision Tree) |
        | | 서포트 벡터 머신(Support Vector Machine) |
        | | k-최근접이웃(K-Nearest Neightbor: KNN) |
        | | 로지스틱 회귀(Logistic Regression) |
        | 회귀분석 | 결정트리(Decision Tree) |
        | | 선형 회귀(Linear Regression) |
        | | 확률적 경사 하강 회귀(Stochastic gradient descent Regression; SGD) |

- **비지도 학습(Unsupervised Learning)**
    - **정의 : 훈련 단계에서 설명변수에 조합에 대응하는 반응변수를 제시하지 않는 학습 방법**
        - 알고리즘은 설명변수의 특징만을 활용하여 목표한 결과를 산출함

    - **주요 알고리즘**

        | 분석 종류 | 알고리즘 |
        |---|---|
        | 군집분석 | K-Means |
        | | 계층적 군집 분석(Hierarchical Cluster Analysis; HCA) |
        | | DBSCAN | 
        | 차원 축소 | 주성분 분석(Principal Component Analysis; PCA) |
        | | 커널 주성분 분석(Kernel Principal Component Analysis) |
        | | 지역적 선형 임베딩(Locally-Linear Embedding; LLE) |
        | | t-SNE(t-distributed Stochastic Neighbor Embedding) |
        | 이상치 탐지 | 가우스 분포 |
        | 연관규칙 | Apriori |
        | | Eclat |

- **준지도 학습(Semi-supervised Learning)**
    - **정의 : 지도 학습과 비지도 학습의 절충안**
        - 모든 설명변수의 조합에 대하여 그에 대응하는 반응변수를 배치할 수 없는 현실을 고려한 학습 방법
        - 레이블(반응변수)이 존재하는 데이터 셋과 존재하지 않는 데이터 셋을 모두 사용함
        - 레이블이 군집 형태에 가까울수록 좋은 결과를 나타냄
    
    - **주요 알고리즘**
        - 심층신뢰신경망(DBN)
        - 제한된 볼츠만 기계(RBM)

- **강화 학습(Reinforcement Learning)**

    ![강화학습](https://tensorflowkorea.files.wordpress.com/2018/05/e18489e185b3e1848fe185b3e18485e185b5e186abe18489e185a3e186ba-2018-05-24-e1848be185a9e1848ce185a5e186ab-12-21-44.png?w=768)

    - **정의 : 행동심리학에서 영감을 받은 학습 방법**
        - 행동에 대한 구체적인 지시 없이 목표만 부여함
        - 보상을 얻기 위한 전략(policy)을 스스로 학습함
        - 환경(environment)을 관찰하고 행동(action)을 실행하여 보상(reward)을 도출함

    - **주요 알고리즘**
        - SARSA
        - Q-Learning

</details>

<details><summary><h3>실시간으로 훈련하는가</h3></summary>

- **온라인 학습**
    - 데이터를 소그룹(mini batch)으로 묶어서 순차적으로 제공하며 모델을 점진적으로 훈련시키는 방법

- **오프라인 학습**
    - 사전에 충분히 훈련된 모델을 사후 갱신 없이 제품에 적용하는 방법
    - 모든 데이터를 한번에 학습함
    - 많은 시간이 소요되고 많은 리소스가 동원되므로 오프라인으로 수행함

</details>

<details><summary><h3>새로운 데이터를 어떠한 방식으로 일반화하는가</h3></summary>

- **사례 기반 학습**
    - 학습된 사례를 기억하는 방식으로 훈련함
    - 새로운 데이터가 들어오는 경우, 학습된 데이터와 새로운 데이터 간 유사도를 측정함
    - 학습된 데이터들 중 유사도가 가장 높은 데이터 유형의 사례로서 새로운 데이터를 분류함

- **모델 기반 학습**
    - **주어진 데이터 셋에 적합한 알고리즘을 채택하여 모델을 설계함**
        - 데이터 셋마다 분석하기에 적합한 알고리즘 및 하이퍼파라미터가 다름
        - 따라서 데이터 셋에 적합한 알고리즘 및 하이퍼파라미터를 찾는 모델 선택 과정이 필요함
    
    - **모델을 훈련하여 주어진 데이터 셋에 가장 적합한 모델 파라미터를 찾아냄**
        - 적합한지 여부는 모델 성능으로 판단함
        - 모델 성능이 얼마나 좋은가는 효용 함수(혹은 적합도 함수)로 평가함
        - 모델 성능이 얼마나 나쁜가는 손실 함수(혹은 비용 함수)로 평가함
        - 일반적으로는 손실 함수를 최소화하는 방향으로 훈련함
    
    - **설계된 모델을 이용하여 새로운 데이터를 어떻게 분류할 것인지 예측함**

</details>

---

## 🛠 [Scikit-Learn Library](https://scikit-learn.org/stable/#)

<details><summary><h3>정의</h3></summary>

- **정의**
    - 머신러닝 기술을 통일된 인터페이스로써 활용할 수 있도록 정리한 라이브러리  
    - 머신러닝 알고리즘, 머신러닝 개발을 위한 프레임워크 및 API 제공  

- **API 사용 방법**    
    1. 적절한 알고리즘 클래스 임포트
    2. 클래스의 하이퍼파라미터를 적절한 값으로 설정하여 인스턴스 생성
    3. 데이터를 피쳐(속성)와 타깃(정답)으로 배치
    4. 인스턴스의 메소드 `fit()`을 통해 인스턴스를 학습용 데이터로 훈련시킴
    5. 인스턴스의 메소드 `predict()`을 통해 훈련된 인스턴스에 테스트용 데이터를 적용함
    
</details>

<details><summary><h3>주요 모듈</h3></summary>

- **알고리즘**    
    
    | 모듈 | 설명 | 예시 |
    |------|------|------|
    | sklearn.tree | 결정 트리 알고리즘 제공 | Decision Tree 등 |
    | sklearn.neighbors | 최근접 이웃 알고리즘 제공 | K-NN 등 |
    | sklearn.svm | 서포트 벡터 머신 알고리즘 제공 |
    | sklearn.naive_bayes | 나이브 베이즈 알고리즘 제공 | 가우시안 NB, 다항 분포 NB 등 |
    | sklearn.cluster | 클러스터링 알고리즘 제공 | K-Means, 계층형 클러스터링, DBSCAN 등 |
    | sklearn.linear_model | 회귀분석 알고리즘 제공 | 선형 회귀, 확률적 경사하강 회귀(SGD), 릿지(Ridge), 라쏘(Lasso), 로지스틱 회귀 등 |
    | sklearn.decomposition | 차원 축소 알고리즘 제공 | PCA, NMF, Truncated SVD 등 |
    | sklearn.ensemble | 앙상블 알고리즘 제공 | Random Forest, AdaBoost, GradientBoost 등 |

- **전처리**
    
    | 모듈 | 설명 | 예시 |
    |------|------|------|
    | sklearn.preprocessing | 데이터 전처리 기능 제공 | 인코더, 스케일러 등 |
    | sklearn.feature_selection | 특성(feature)을 선택할 수 있는 기능 제공 | 
    | sklearn.feature_extraction | 특성(feature)을 추출할 수 있는 기능 제공 |
    | sklearn.pipeline | 특성 처리, 학습, 예측을 묶어서 실행할 수 있는 기능 제공 |

- **검증 및 성능 평가 지표**

    | 모듈 | 설명 | 예시 |
    |------|------|------|
    | sklearn.model_selection | 교차 검증, 최적 하이퍼파라미터 추출 API 제공 | GridSearch 등 |
    | sklearn.metrics | 성능 평가 지표 제공 | Accuracy, Precision, Recall, ROC-AUC, RMSE 등 |

</details>

<details><summary><h3>sklearn.datasets</h3></summary>

- **내장 데이터 형식**

    | 이름 | 설명 |
    |------|------|
    | DESCR | 자료에 대한 설명 |
    | data | 설명 변수 |
    | target | 반응 변수 |
    | feature_names | 설명 변수 이름 리스트 |
    | target_names | 반응 변수 이름 리스트 |    
    
- **내장 데이터 셋 목록**

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

<details><summary><h3>권장 사항</h3></summary>

![이미지](https://scikit-learn.org/stable/_static/ml_map.png)

</details>
