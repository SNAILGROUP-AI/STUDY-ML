## 😈 차원의 저주

- **차원의 저주(The curse of dimensionality)**
    - 정의 : 고차원 공간에서 데이터를 분석하고 정리할 때 발생하는 다양한 현상
    - 현상 : 차원이 증가하면 공간의 부피가 너무 빨리 증가하여 사용 가능한 데이터가 상대적으로 희소해짐
    - 문제점 : 개별 차원 내 학습할 데이터 개수가 희소해져서 학습이 제대로 이루어지지 않음
    - 해법 : 차원 축소

- **차원 축소(Dimensionality Reduction)**
    - **정의 : 데이터 세트의 차원, 곧 설명변수의 개수를 줄이는 작업**

    - **차원 축소 방법**
        - **차원 선별**
            - 차원 선별(Feature Selection)

        - **차원 추출**
            - 선형 차원 축소(Matrix Factor or Projection)
            - 비선형 차원 축소(Neighbor Graphs or Manifold Learning)

    - **지도학습 : 차원 선별 방식**
        - 반응변수에 관한 정보를 바탕으로 유의미하지 않은 설명변수를 선별하여 제거함
        - 분류분석 : 승산비 기준 유의미하지 않은 설명변수 제거
        - 선형회귀분석 : 선형회귀분석의 4개 가정 하에 분산팽창계수 기준 다중공선성이 낮지 않은 설명변수 제거
    
    - **비지도학습 : 차원 추출 방식**
        - 반응변수에 관한 정보가 없으므로 지도학습의 차원 선별 기법들을 사용할 수 없음
        - 따라서 다수의 차원들을 대표할 수 있는 주요한 성분을 추출함으로써 차원을 축소해야 함

---

## 🧚‍♀️ 주성분 분석(Principle Component Analysis; PCA)

<details><summary><h3>주성분 분석이란 무엇인가</h3></summary>

- **사영(Projection)**

    ![은하](https://t1.daumcdn.net/cfile/tistory/99CB343359F2DA5E07)

    - **정의 : 선형 차원 축소 기법**
        - 고차원 입체의 형태를 가장 잘 나타낼 수 있는 저차원 단면을 찾는 행위
    
    - **종류**
        - 주성분 분석(Principle Component Analysis; PCA)
        - 선형 판별 분석(Linear Discriminant Analysis; LDA)
        - LDA(Linear Discriminant Analysis)
        - NMF(Non-Negative Matrix Factorization)

- **주성분 분석(Principle Component Analysis)**

    - **정의**
        - 변수 간에 존재하는 상관관계를 이용하여 이를 대표하는 주성분을 추출하여 차원을 축소하는 기법

    - **이슈**

        ![04AD38E4-3544-4BEC-952C-0B4542AA1538](https://user-images.githubusercontent.com/116495744/224222113-e15b8091-9a64-4a49-bd7d-916d4bb75874.jpg)

        - 정보(특성) 유실 문제

    - **해법**

        ![IMG_7017](https://user-images.githubusercontent.com/116495744/224222115-02d0ecb3-112d-4417-a39f-8d69f91ad84f.jpg)

        - 분산을 최대한 보존함으로써 레코드 간 특성별 차이를 보존함

</details>

<details><summary><h3>주성분 분석의 이해</h3></summary>

- **주성분 분석의 직관적 이해**

    - **Whitening**

        ![IMG_7004](https://user-images.githubusercontent.com/116495744/224222107-98d84b92-79bd-47c0-b430-aa2584b9e22f.JPG)

        - N개의 설명변수에 대하여 모든 설명변수의 평균을 원점으로 하는 N차원 그래프를 생성함
        - 데이터 세트를 그래프에 묘사함

    - **주성분 추출**

        ![사영](https://user-images.githubusercontent.com/116495744/224226095-898ac9a8-9cec-4b0d-a553-074bbc6a1ffd.jpeg)

        - 원점을 지나는 직선 중에서 모든 레코드를 사영했을 때 SS가 가장 큰 직선을 찾음
        - 원점을 지나고 앞서 구한 직선과 직교하면서 SS가 가장 큰 직선을 찾음
        - 원점을 지나고 앞서 구한 직선들과 직교하면서 SS가 가장 큰 직선을 찾음
        - 위 과정을 반복하면서 차원의 갯수만큼의 직선을 찾음

    - **주성분 선별**
        - **직선**
            - 위 절차를 통해 찾은 직선들을 해당 데이터 세트의 주성분(Principle Component; PC)이라고 정의함
        
        - **SS(Sum of Squared Distance)**
            - 원점과 사영점 간 거리 제곱의 합을 해당 직선의 SS라고 정의함
            - 원점과 특정 레코드의 사영점 간 거리의 제곱을 해당 레코드의 주성분값으로 해석함
            - 전체 직선의 SS 대비 특정 직선의 SS를 해당 직선이 전체 특성을 설명하는 정도로 해석함
        
        - **주성분 선별**
            - N차원 데이터 세트를 k차원으로 줄이고자 하는 경우
            - SS 기준 상위 k개 주성분을 추출함

- **주성분 분석의 수학적 이해**
    
    - **주성분 추출**
        - 데이터 세트의 공분산행렬을 구함
        - 공분산행렬의 고유벡터와 고유값을 구함
    
    - **주성분 선별**
        - **고유벡터(EigenVector)**
            - 위 절차를 통해 찾은 고유벡터를 해당 데이터 세트의 주성분이라고 정의함
            - 특정 레코드에 대응하는 고유벡터의 원소를 해당 레코드의 주성분값으로 해석함
        
        - **고유값(EigenValue)**
            - 전체 고유벡터의 고유값 대비 특정 고유벡터의 고유값을 해당 고유벡터가 전체 특성을 설명하는 정도로 해석함
        
        - **주성분 선별**
            - 고유벡터를 고유값 기준으로 내림차순 정렬
            - 원하는 차원 수만큼 고유벡터를 선별

</details>

<details><summary><h3>주요 개념</h3></summary>

- **분산(Variance; Var)**

    $$var(X) = \displaystyle\sum_{i=0}^{n}\frac{(X-\overline{X})^2}{n}$$

    - 정의 : 단차원 데이터 세트에 대하여 평균점을 중심으로 레코드가 흩어진 정도

- **공분산(Covariance; Cov)**

    $$cov(X, Y) = \displaystyle\sum_{i=0}^{n}\frac{(X_i-\overline{X})(Y_i-\overline{Y})}{n}$$

    - 정의 : 다차원 데이터 세트에 대하여 평균점을 중심으로 레코드가 흩어진 정도
    - 해석 : 2개의 축을 가정했을 때, 한 변수의 증감에 따른 다른 변수의 증감 경향성

- **공분산행렬(Covariance Matrix)**

    $$ \sum = 
    \begin{pmatrix}
    var(X) & cov(X, Y) \\
    cov(Y, X) & var(Y)
    \end{pmatrix} $$

    - **정의**
        - 다차원 데이터 세트를 구성하는 변수(혹은 축) $X, Y, Z, \cdots$ 에 대하여
        - $i$ 번째, $j$ 번째 변수(혹은 축)의 공분산을 $(i, j)$ 의 값으로 가지는 정방행렬

    - **상관관계와 공분산행렬**
        - **상관행렬(Correlation Matrix)** : 공분산행렬을 정규화한 행렬
        - **피어슨 상관계수(Pearson Correlation Coefficient)** : 상관행렬을 구성하는 스칼라

    - **선형변환과 공분산행렬**

        ![공분산행렬과 고유벡터](https://user-images.githubusercontent.com/116495744/224226188-05975c29-4ac8-4572-b796-fb7eec3bab5a.jpeg)

        - 임의의 행렬 P에 대하여 그 공분산행렬을 행렬 Q에 내적하는 경우
        - 그래프상으로 표현된 Q의 분포가 P의 분포와 유사한 형태로 변환됨

- **고유벡터(EigenVector)와 고유값(EigenValue)**

    $$\sum \cdot V = \lambda \times V$$

    - **고유벡터(EigenVector)** : 임의의 데이터 세트에 대하여 그 공분산행렬을 내적하여 선형변환하더라도 방향이 변환 전과 동일한 벡터
    - **고유값(EigenValue)** : 임의의 데이터 세트에 대하여 그 공분산행렬을 내적하기 전 고유벡터의 길이 대비 내적한 후 고유벡터의 길이
    
</details>

<details><summary><h3>SK-Learn의 주성분 분석 알고리즘</h3></summary>

- **사용 방법**

    ```
    from sklearn.decomposition import PCA

    # PCA 알고리즘 인스턴스 생성
    # 축소할 차원의 수를 3으로 설정
    pca = PCA(n_components = 3)

    # 주성분 탐색
    pca.fit(X)

    # 데이터 세트 차원 축소
    X = pca.transform(X)
    ```

- **주요 하이퍼파라미터**
    - `random_state = None`
    - `n_components` : 축소할 차원의 개수
    - `whiten = False` : 원점을 모든 설명변수들의 평균으로 조정할 것인지 여부

- **다음의 속성을 통해 훈련된 모델의 정보를 확인할 수 있음**
    - `n_samples_` : 레코드 개수
    - `n_features_` : 축소 전 차원의 개수
    - `feature_names_in_` : 축소 전 차원명
    - `mean_` : 축소 전 차원별 평균
    - `n_components_` : 축소 후 차원의 개수
    - `components_` : 고유벡터
    - `explained_variance_` : 각 고유벡터의 고유값
    - `explained_variance_ratio_` : 전체 고유벡터의 고유값 대비 각 고유벡터의 고유값

</details>

---

## t-SNE(t-distributed Stochastic Neighbor Embedding)

<details><summary><h3>t-SNE 란 무엇인가</h3></summary>

- **다양체 학습(Manifold Learning)**

    ![IMG_355193D3C896-1](https://user-images.githubusercontent.com/116495744/224497076-8a2e6100-88a5-444c-abb9-377e61e961ee.jpeg)

    - **정의 : 비선형 차원 축소 기법**
        - **다양체(Manifold)** : 데이터 세트를 고차원 공간에 묘사했을 때, 그 레코드들을 잘 아우를 수 있는 저차원 공간(SubSpace)
        - **다양체 학습(Manifold Learning)** : 데이터 세트를 잘 아우를 수 있는 다양체를 찾아 해당 데이터 세트의 차원을 축소하는 기법

    - **종류**
        - t-SNE(t-distributed Stochastic Neighbor Embedding)
        - LLE(Locally Linear Embedding)
        - ISOMAP
        - MDS(Multi-Dimensioning Scaling)
        - AE(Auto Encoder)

- **t-SNE(t-distributed Stochastic Neighbor Embedding)**
    - **정의**
        - 고차원 공간에서 인접한(Neighbor) 두 벡터가 저차원 공간에서도 인접하도록 고차원에서의 유사도를 보존하며 차원을 축소하는 방법
    
    - **방법**
        - 데이터 세트를 고차원 공간에 묘사함
        - 레코드 i, j에 대하여, 고차원 공간에서 i, j 간 거리의 기대값 $p$ 를 계산함
        - 저차원 공간에서 i, j 간 거리의 기대값 $q$ 를 계산함
        - $p$, $q$ 의 차이를 반영하는 손실함수 $C(p, q)$ 를 정의함
        - 손실을 최소화하는 저차원 공간을 해당 데이터 세트의 다양체로 정의함
        - 데이터 세트의 차원을 다양체로 변환함

</details>

<details><summary><h3>SK-Learn의 t-SNE 알고리즘</h3></summary>

- **사용 방법**

    ```
    from sklearn.manifold import TSNE

    tsne = TSNE()

    X = tsne.fit_transform(X)
    ```

- **주요 하이퍼파라미터**

</details>

---

## 📝 Practice

- [**실습 코드**]()