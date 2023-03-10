## 차원의 저주

- **차원의 저주(The curse of dimensionality)**
    - 정의 : 고차원 공간에서 데이터를 분석하고 정리할 때 발생하는 다양한 현상
    - 현상 : 차원이 증가하면 공간의 부피가 너무 빨리 증가하여 사용 가능한 데이터가 상대적으로 희소해짐
    - 문제점 : 개별 차원 내 학습할 데이터 개수가 희소해져서 학습이 제대로 이루어지지 않음
    - 해법 : 차원 축소

- **차원 축소(dimensionality reduction)**
    - **정의 : 데이터 세트의 차원, 곧 설명변수의 개수를 줄이는 작업**

    - **지도학습에서의 차원 축소 : 설명변수와의 상관관계를 가정한 상태에서 진행됨**
        - 분류분석 : 승산비 기준 유의미하지 않은 설명변수 제거
        - 선형회귀분석 : 선형회귀분석의 4개 가정 하에 분산팽창계수 기준 다중공선성이 낮지 않은 설명변수 제거
    
    - **비지도학습에서의 차원 축소**
        - 설명변수에 관한 정보가 없으므로 지도학습 시 차원 축소 기법들을 사용할 수 없음
        - 따라서 다수의 차원들을 대표할 수 있는 주요한 성분들을 추출함으로써 차원을 축소해야 함

---

## 주성분 분석

<details><summary><h3>주성분 분석(Principle Component Analysis; PCA)</h3></summary>

- **주성분 분석(Principle Component Analysis)**

    - **정의 : 변수 간에 존재하는 상관관계를 이용하여 이를 대표하는 주성분을 추출하여 차원을 축소하는 기법**

    - **이슈 : 정보(특성) 유실 문제**

        ![04AD38E4-3544-4BEC-952C-0B4542AA1538](https://user-images.githubusercontent.com/116495744/224222113-e15b8091-9a64-4a49-bd7d-916d4bb75874.jpg)

    - **해법 : 분산을 최대한 보존함으로써 레코드 간 특성별 차이를 보존함**

        ![IMG_7017](https://user-images.githubusercontent.com/116495744/224222115-02d0ecb3-112d-4417-a39f-8d69f91ad84f.jpg)

</details>

<details><summary><h3>직관적 이해</h3></summary>

- **Whitening**

    ![IMG_7004](https://user-images.githubusercontent.com/116495744/224222107-98d84b92-79bd-47c0-b430-aa2584b9e22f.JPG)

    - N개의 설명변수에 대하여 모든 설명변수의 평균을 원점으로 하는 N차원 그래프를 생성함
    - 데이터 세트를 그래프에 묘사함

- **주성분 추출**

    ![29571075-D199-4C47-8A70-DCC02FF2458B](https://user-images.githubusercontent.com/116495744/224223522-5b2f4407-c8af-40e8-bce1-fa58fd49bb50.jpg)

    - 원점을 지나는 직선 중에서 모든 레코드를 사영했을 때 SS가 가장 큰 직선을 찾음
    - 원점을 지나고 앞서 구한 직선과 직교하면서 SS가 가장 큰 직선을 찾음
    - 원점을 지나고 앞서 구한 직선들과 직교하면서 SS가 가장 큰 직선을 찾음
    - 위 과정을 반복하면서 차원의 갯수만큼의 직선을 찾음

- **주성분 선별**
    - **주성분(Principle Component; PC)**
        - 위 절차를 통해 찾은 직선들을 해당 데이터 세트의 주성분이라고 정의함
    
    - **SS(Sum of Squared Distance)**
        - 원점과 사영점 간 거리 제곱의 합을 해당 주성분의 SS라고 정의함
        - 원점과 특정 레코드의 사영점 간 거리의 제곱을 해당 레코드의 주성분값으로 해석함
        - 전체 주성분의 SS 대비 특정 주성분의 SS를 해당 직선이 전체 특성을 설명하는 정도로 해석함
    
    - **주성분 선별**
        - N차원 데이터 세트를 k차원으로 줄이고자 하는 경우
        - SS 기준 상위 k개 주성분을 추출함

</details>

<details><summary><h3>수학적 이해</h3></summary>

- **주요 개념**
    - **분산(Variance; Var)**
    
        - 정의 : 단차원 데이터 세트에 대하여 평균점을 중심으로 레코드가 흩어진 정도
    
    - **공분산(Covariance; Cov)**

        - 정의 : 다차원 데이터 세트에 대하여 평균점을 중심으로 레코드가 흩어진 정도
        - 해석 : 2개의 축을 가정했을 때, 한 확률변수의 증감에 따른 다른 확률변수의 증감 경향성
    
    - **공분산행렬(Covariance Matrix)**

        - 정의 : 분산
 
        - **상관관계와 공분산행렬**
            - **상관행렬(Correlation Matrix)** : 공분산행렬을 정규화한 행렬
            - **피어슨 상관계수(Pearson Correlation Coefficient)** : 상관행렬을 구성하는 스칼라
 
        - **선형변환과 공분산행렬**

            ![공분산행렬과 고유벡터](https://user-images.githubusercontent.com/116495744/224226188-05975c29-4ac8-4572-b796-fb7eec3bab5a.jpeg)

            - 임의의 행렬 P에 대하여 그 공분산행렬을 행렬 Q에 내적하는 경우
            - 그래프상으로 표현된 Q의 분포가 P의 분포와 유사한 형태로 변환됨

    - **고유벡터(EigenVector)**
    
    - **고유값(EigenValue)**

![사영](https://user-images.githubusercontent.com/116495744/224226095-898ac9a8-9cec-4b0d-a553-074bbc6a1ffd.jpeg)

</details>

---

## SK-Learn의 주성분 분석 알고리즘

<details><summary><h3>사용 방법</h3></summary>

- **사용 방법**

    ```
    from sklearn.decomposition import PCA
    ```

- **주요 하이퍼파라미터**

- **다음의 속성을 통해 훈련된 모델의 정보를 확인할 수 있음**
    - `n_samples_` : 레코드 개수
    - `n_features_` : 축소 전 차원의 개수
    - `feature_names_in_` : 축소 전 차원명
    - `mean_` : 축소 전 차원별 평균
    - `n_components_` : 축소 후 차원의 개수
    - `explained_variance_`          # 첫 번째 축, 두 번째 축으로 캡쳐했을 때 데이터의 분산이었어
    - `explained_variance_ratio_`    # 첫 번째 축으로 92%, 두 번째 축으로 5% 데이터를 보존했어
    - `components_`                  # 고유행렬 : 원본데이터 기준으로 가장 많은 데이터 담고 있는 첫 번째, 두 번째 축의 고유행렬(eigen vector)

</details>

---

## 📝 Practice

- [**실습 코드**]()