# What? Machine Learning

---

## 정의

<details><summary><h3>정의</h3></summary>

- 정의
    - 컴퓨터 프로그램이 주어진 데이터를 통해 데이터 처리 경험을 훈련함으로써 정보 처리 능력을 향상시키는 행위
    - 설명변수 X와 반응변수 Y에 대하여, 두 변수 간 상관관계를 서술한 수식을 찾는 능력을 향상시키는 행위

- 알고리즘과 모델
    - 알고리즘 : 어떠한 문제를 해결하기 위한 일련의 절차나 방법
    - 모델 : 상관관계를 식으로 표현한 것으로서 알고리즘을 통해 도출된 설명변수와 반응변수의 관계를 나타낸 함수

</details>

<details><summary><h3>장점</h3></summary>

- 전통적인 방식으로는 해결할 수 없는 복잡한 문제를 쉽게 풀 수 있음
    - 전통적인 방식의 경우 개발자가 직접 규칙을 설계하고 파라미터를 조정해야 함
    - 새로이 등장하는 문제들을 전통적인 방식으로 푸는 경우 설계할 규칙이나 조정할 파라미터가 복잡하고 다양함
    - 머신러닝의 경우 개발자가 적절한 알고리즘을 채택하고 적절한 파라미터를 설정하면 모델이 스스로 방법을 찾아냄
    - 따라서 전통적인 방식에 비해 머신러닝으로 문제를 풀 경우 개발자가 직접 수행해야 하는 업무가 줄어듦

- 유동적인 환경에 대처할 수 있음
    - 대용량 데이터로부터 스스로 규칙을 찾아내어 복잡한 문제에 대응함
    - 따라서 학습되지 않은 데이터에 대해서도 적용할 수 있음

</details>

<details><summary><h3>구분</h3></summary>

![머신러닝 구분](https://github.com/trekhleb/homemade-machine-learning/blob/master/images/machine-learning-map.png?raw=true)

</details>

<details><summary><h3>WorkFlow</h3></summary>

![머신러닝워크플로우](https://content.altexsoft.com/media/2017/04/Screenshot_3.png)

- Collect data  : 유용한 데이터를 최대한 많이 확보하고 하나의 데이터 세트로 통합함

- Prepare data  : 결측값, 이상값, 기타 데이터 문제를 적절하게 처리하여 사용 가능한 상태로 준비함

- Split data :  데이터 세트를 학습용 세트와 평가용 세트로 분리함

- Train a model : 학습용 데이터 세트의 일부를 통해 모델이 데이터 내 패턴을 찾도록 훈련함

- Validate a model : 학습용 데이터 세트의 나머지를 통해 모델이 데이터 내 패턴을 잘 찾아냈는지 확인함

- Test a model : 평가용 데이터 세트를 통해 모델의 성능을 파악함

- Deploy a model :  모델을 의사결정 시스템에 탑재함

- Iterate :  새로운 데이터를 확보하고 모델에 적용하여 모델을 점진적으로 개선해나감

</details>

---

<details><summary><h2>분류</h2></summary>

### 사람의 감독 하에 훈련하는가

- 지도 학습(Supervised Learning)
    - 훈련 단계에서 설명변수의 조합에 대응하는 반응변수를 함께 제시하는 학습 방법
        - 일고리즘은 설명변수와 반응변수의 상관관계를 가장 잘 설명할 수 있는 모델을 찾음
        - 알고리즘은 모델을 사용하여 새로운 설명변수에 대하여 예측을 수행함
    
    - 주요 알고리즘

        | 분석 종류 | 알고리즘 |
        |---|---|
        | 분류분석 | 결정트리(Decision Tree) |
        | | 서포트 벡터 머신(Support Vector Machine) |
        | | k-최근접이웃(K-Nearest Neightbor: KNN) |
        | | 로지스틱 회귀(Logistic Regression) |
        | 회귀분석 | 결정트리(Decision Tree) |
        | | 선형 회귀(Linear Regression) |
        | | 확률적 경사 하강 회귀(Stochastic gradient descent Regression; SGD) |

- 비지도 학습(Unsupervised Learning)
    - 훈련 단계에서 설명변수에 조합에 대응하는  반응변수를 제시하지 않는 학습 방법
        - 알고리즘은 설명변수의 특징만을 활용하여 목표한 결과를 산출함

    - 주요 알고리즘

        | 분석 종류 | 알고리즘 |
        |---|---|---|
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

- 준지도 학습(Semi-supervised Learning)
    - 지도 학습과 비지도 학습의 절충안
        - 모든 설명변수의 조합에 대하여 그에 대응하는 반응변수를 배치할 수 없는 현실을 고려한 학습 방법
        - 레이블(반응변수)이 존재하는 데이터 셋과 존재하지 않는 데이터 셋을 모두 사용함
    
    - 레이블이 군집 형태에 가까울수록 좋은 결과를 나타냄
    
    - 주요 알고리즘
        - 심층신뢰신경망(DBN)
        - 제한된 볼츠만 기계(RBM)

- 강화 학습(Reinforcement Learning)

    ![강화학습](https://tensorflowkorea.files.wordpress.com/2018/05/e18489e185b3e1848fe185b3e18485e185b5e186abe18489e185a3e186ba-2018-05-24-e1848be185a9e1848ce185a5e186ab-12-21-44.png?w=768)
https://tensorflow.blog/%ed%95%b8%ec%a6%88%ec%98%a8-%eb%a8%b8%ec%8b%a0%eb%9f%ac%eb%8b%9d-1%ec%9e%a5-2%ec%9e%a5/1-3-%eb%a8%b8%ec%8b%a0%eb%9f%ac%eb%8b%9d-%ec%8b%9c%ec%8a%a4%ed%85%9c%ec%9d%98-%ec%a2%85%eb%a5%98/

    - 행동심리학에서 영감을 받은 학습 방법
    
    - 보상을 얻기 위한 전략(policy)을 스스로 학습함
        - 행동에 대한 구체적인 지시 없이 목표만 부여함
        - 환경(environment)을 관찰하고 행동(action)을 실행하여 보상(reward)을 도출함

    - 주요 알고리즘
        - SARSA
        - Q-Learning

- 전이 학습(Transfer Learning)
    - 새로운 문제를 해결하고자 할 때 기존에 학습된 모델을 이용하여 새로운 모델을 만드는 방법
        - 기존 학습 방법은 훈련용 데이터와 실제 분석하려는 데이터가 유사한 분포를 가지고 있다는 가정에 기초함
        - 때문에 가정이 성립하지 않는 문제에 대해서는 좋은 결과를 보기 어려움
        - 이러한 경우 pre-trained model을 새로운 문제에 알맞게 조정하여 사용하면 성능을 높일 수 있음
    
    - 다음의 상황에서 활용하기에 적절함
        - 잘 훈련되어 있는 모델이 존재하는 경우
        - 해결하고자 하는 문제가 기존 모델이 해결 가능한 문제와 유사한 경우
        - 훈련용 데이터가 부족한 경우

### 실시간으로 갱신되는 데이터를 통해 점진적으로 훈련하는가
- 온라인 학습
    - 데이터를 소그룹(mini batch)으로 묶어서 순차적으로 제공하며 점진적으로 훈련시키는 방법

- 오프라인 학습
    - 사전에 충분히 훈련된 시스템을 사후 갱신 없이 제품에 적용하는 방법
    - 모든 데이터를 사용하여 학습함
    - 많은 시간이 소요되고 많은 리소스가 동원되므로 오프라인으로 수행함

### 무엇을 기반으로 훈련하는가
- 사례 기반 학습
- 모델 기반 학습

</details>

---

<details><summary>이슈</summary>

### 데이터
- 충분하지 않은 양의 데이터를 통한 학습
- 대표성이 없는 데이터를 통한 학습
- 품질이 낮은 데이터를 통한 학습
- 반응변수와의 연관성이 낮은 설명변수를 통한 학습

### 과대적합/과소적합
![과대적합과 과소적합](https://tensorflowkorea.files.wordpress.com/2017/06/fig2-01.png?w=640)

- 모델이 훈련 시 제공되는 데이터에 과대 혹은 과소 적합되는 경우
- 즉, 모델이 새로운 사례에 대하여 일반화되지 않는 경우

</details>

---

<details><summary>다른 분야와의 관계</summary>

### AI, ML, DL
- 인공지능(Artificial Intelligence; AI) : 사람처럼 학습하고 추론할 수 있는 시스템을 만드는 기술

- 머신러닝(Machine Learning; ML) : 규칙을 프로그래밍하지 않아도 주어진 데이터에서 자동으로 규칙을 발견하는 기술

- 딥러닝(Deep Learning; DL) : 인공 신경망을 기반으로 하는 머신러닝 기술

### 빅데이터와 머신러닝의 관계
- 빅데이터(big data) : 기존의 데이터베이스로는 수집, 처리, 저장, 분석을 수행하기 어려울 만큼 방대한 양의 데이터

- 빅데이터 시스템(big data system) : 빅데이터를 다루기 위한 시스템

- 빅데이터 엔지니어링(big data engineering) :빅데이터를 다루는 방법

- 빅데이터와 머신러닝의 관계
    - 본래 빅데이터의 개념은 데이터베이스에서 기원하여 머신러닝과는 별개로 발전해왔음
    - 학습 가능한 데이터의 양이 머신러닝 모델의 성능을 좌우하게 되면서, 오늘날 머신러닝 분야에서 유의미해짐

</details>

---

