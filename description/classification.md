## 🤓 분류분석이란 무엇인가

- **정의 : 데이터를 나누는 발산 작업**
    - 대상의 정체를 규명하는 작업
    - 데이터를 일정한 기준에 따라 몇 가지 범주로 나누는 작업
    - 반응변수가 범주형 변수인 경우

- **학습 양상**
    - **학습** : 주어진 데이터를 잘 분류할 수 있는 함수를 찾는 행위
    - 함수의 형태는 수식이 될 수도 있고, 규칙이 될 수도 있음
    - 이상적인 분류기는 새로운 데이터의 정체를 잘 규명할 수 있음

- **구분**
    - **이항분류분석(binary classification)** : 대상을 둘 중 하나로 분류하는 분석 방법
    - **다항분류분석(multi-category classificaiton)** : 대상을 3개 이상의 범주 중 하나로 분류하는 분석 방법

---

## 🌳 결정 트리(Decision Tree)

<details><summary><h3>결정 트리란 무엇인가</h3></summary>

- **정의 : 데이터에 내재된 규칙을 발견하여 수형도 기반의 분류 규칙을 세우고 데이터를 분류하는 알고리즘**
    
- **주요 이슈 : 트리를 어떻게 분할할 것인가**
    - 가지를 몇 번 뻗을 것인가
    - 한 범주당 데이터가 몇 개 남았을 때 가지치기를 멈출 것인가
    
- **주의 사항**
    - node가 깊어질수록 성능이 저하될 수 있음
    - 범주마다 균일한 데이터 세트를 구성할 수 있도록 하이퍼파라미터를 설정해야 함

</details>

<details><summary><h3>균일도</h3></summary>

- **정의 : leaf node에 각 범주에 해당하는 데이터만 포함되어 있는가**

- **예시**
    - `color`을 기준으로 바둑알을 구분한다고 가정하자
    - 범주로는 `black` , `white` 가 존재함
    - 범주 `black`에 검정색 바둑알만 포함되어 있다면 균일도가 높다고 해석함
    - 범주 `black`에 흰색 바둑알이 많이 섞여 있을수록 균일도가 낮다고 해석함

- **decision node**
    - 균일도가 높은 데이터 세트를 먼저 분류할 수 있도록 규칙을 구성함
    - 즉, 균일도를 높이는 방향으로 가지치기를 진행함

- **균일도 측정 방법**
    - **지니 불순도** : 경제학에서 불평등 정도를 나타내는 지수를 활용하여 균일도를 측정하는 방법
    - **엔트로피 불순도** : 열역학에서 물체의 혼잡한 정도를 나타내는 지수를 활용하여 균일도를 측정하는 방법

</details>

<details><summary><h3>결정 트리의 구조</h3></summary>

![아이리스 결정트리 예시](https://user-images.githubusercontent.com/116495744/221340236-6c4043c6-6b30-4af2-9e7f-cfe79b00371a.png)

- **root node** : 최상위 노드

- **decision node** : 규칙 노드

- **leaf node** : 최종 범주

- **gini** : 데이터 분포의 균일도

- **samples** : 임의의 규칙에 대하여 해당 규칙을 만족하는 데이터 건수

- **value** : 각 범주의 데이터 건수

</details>

<details><summary><h3>SK-Learn의 결정 트리 알고리즘</h3></summary>

- **사용 방법**

    ```
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    # 결정 트리 알고리즘 인스턴스 생성
    dt_clf = DecisionTreeClassifier()
    
    # 훈련용 데이터 세트를 통해 인스턴스를 훈련시켜서 모델 설계
    dt_clf.fit(X_train, y_train)

    # 평가용 데이터 세트를 통해 예측
    y_predict = dt_clf.predict(X_test)

    # 대표적인 성능 평가 지표인 결정계수를 통해 성능 평가
    score = accuracy_score(y_test, y_predict)
    print(score)
    ```

- **주요 하이퍼파라미터**
    - `random_state = None`
    
    - `criterion = 'gini'` : 균일도 측정 방법
        - `gini` : 지니 불순도
        - `entropy` : 엔트로피 불순도
    
    - `max_nodes = None` : 트리 최대 깊이
    
    - `max_features = None` : decision node에서 최적 분할을 위해 고려되어야 할 설명변수의 최대 개수
    
    - `min_samples_split = 2` : 특정 노드에서 하위 노드로 가지치기 하기 위한 최소한의 샘플 개수
    
    - `min_samples_leaf` : leaf node가 되기 위한 최소한의 샘플 개수

- **다음을 통해 훈련된 모델의 정보를 시각화할 수 있음**

    ```
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    # 설명변수 이름이 담긴 리스트 생성
    feature_list = []

    # plot tree 크기 설정
    plt.figure(figsize = (20, 20))
    
    # plot tree 생성
    plot_tree(
        dt_clf, 
        filled = True, 
        fontsize = 14, 
        feature_names = feature_list
        )
    
    # plot tree 출력
    plt.show()
    ```

</details>

---

## 👫 최근접 이웃(k-Nearest Neighbors; k-NN)

<details><summary><h3>최근접 이웃이란 무엇인가</h3></summary>

- **정의 : 기하학적 거리를 규칙으로 하여 데이터를 분류하는 알고리즘**
    - 임의의 설명변수 조합이 나타내는 좌표평면 상의 한 점에 대하여,
    - 해당 점과 가장 가깝게 위치하는 점이 의미하는 설명변수 조합의 범주로 분류함

- **주요 이슈 : 참조할 이웃의 개수를 얼마로 설정할 것인가**

    ![최근접이웃](https://miro.medium.com/max/405/0*QyWp7J6eSz0tayc0.png)

</details>

<details><summary><h3>SK-Learn의 최근접 이웃 알고리즘</h3></summary>

- **사용 방법**

    ```
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    # 최근접 이웃 알고리즘 인스턴스 생성
    knn_clf = KNeighborsClassifier()
    
    # 훈련용 데이터 세트를 통해 인스턴스를 훈련시켜서 모델 설계
    knn_clf.fit(X_train, y_train)

    # 평가용 데이터 세트를 통해 예측
    y_predict = knn_clf.predict(X_test)

    # 대표적인 성능 평가 지표인 결정계수를 통해 성능 평가
    score = accuracy_score(y_test, y_predict)
    print(score)
    ```

- **주요 하이퍼파라미터**
    - `metric = 'minkowsi'` : 거리 측정 방법
        - `minkowsi` : 유클리디안 거리 측정 방법과 맨해튼 거리 측정 방법을 일반화한 측정 방법
        - `euclidean` : 유클리디안 거리 측정 방법
        - `manhattan` : 맨해튼 거리 측정 방법

    - `p = 2` : `metric`이 `minkowsi`인 경우 추가 설정하는 하이퍼파라미터
        - `1` : 맨해튼 거리 측정 방법
        - `2` : 유클리디안 거리 측정 방법

    - `n_jobs = None` : 이웃을 검색하기 위해 병렬로 작업하는 코어의 개수
        - `-1` : 모든 코어를 동원함
    
    - `n_neighbors = 5` : 참조할 이웃의 개수
    
    - `weights = 'uniform'` : 가중치 부여 방법
        - `uniform` : 각 이웃에 동일한 가중치를 부여함
        - `distance` : 거리가 가까울수록 더 큰 가중치를 부여함

</details>

---

## 👥 로지스틱 회귀(Logistic Regression)

<details><summary><h3>로지스틱 회귀란 무엇인가</h3></summary>

- **정의 : 경사하강법에 근거하여 도출된 회귀식을 활용하는 이항분류분석 알고리즘**

- **로지스틱 회귀식의 수학적 이해**

    ![시그모이드 함수](https://user-images.githubusercontent.com/116495744/221402155-596e45c2-5d0d-40a6-ae23-9589b48f807c.png)

    - **연결함수** : 회귀식의 결과값을 이항범주로 변환하는 함수
        ### $$y=f(x)=b+wX$$
        - $y$ : 이항 반응변수
        - $f(x)$ : 연결함수(Link Function)
        - $b+wX$ : 회귀식
    
    - **승산(odds)** : 이항범주를 성패로 정의할 때, 1번 실패할 때 성공할 횟수
        ### $$odds=\frac{p}{1-p}$$
        - $p$ : 성공할 확률
        - $1-p$ : 실패할 확률
    
    - **로짓(logist + Probit)** : 승산에 대하여 자연로그를 취한 값
        ### $$ln(\frac{p}{1-p})$$
        - 로짓 변환은 승산의 범위를 선형 회귀식의 범위와 일치시키기 위한 작업임
        ### $$0 \lt p\lt 1$$
        - 성공할 확률 p는 0~1 사이의 값을 가짐
        ### $$0 \lt \frac{p}{1-p} \lt \infty$$
        - 승산은 양의 실수 범위 값을 가짐
        ### $$\infty \lt ln(\frac{p}{1-p}) \lt \infty$$
        - 로짓은 음의 무한대에서 양의 무한대까지 실수 전범위 값을 취할 수 있음
    
    - **로짓(logit) 함수** : 연결함수를 로짓으로 가지는 선형 회귀식
        ### $$ln({\frac{p}{1-p}})=b+wX$$

    - **시그모이드(sigmoid) 함수** : 로짓 함수의 역함수로서 로지스틱 함수라고도 부름
        ### $$p=\frac{e^{b+wX}}{1+e^{b+wX}}$$

</details>

<details><summary><h3>SK-Learn의 로지스틱 회귀 알고리즘</h3></summary>

- **사용 방법**

    ```
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # 로지스틱 회귀 알고리즘 인스턴스 생성
    log_reg = LogisticRegression()
    
    # 훈련용 데이터 세트를 통해 인스턴스를 훈련시켜서 모델 설계
    log_reg.fit(X_train, y_train)

    # 평가용 데이터 세트를 통해 예측
    y_predict = log_reg.predict(X_test)

    # 대표적인 성능 평가 지표인 결정계수를 통해 성능 평가
    score = accuracy_score(y_test, y_predict)
    print(score)
    ```

- **주요 하이퍼파라미터**
    - `random_state = None`
    
    - `penalty = l2` : 가중치 규제 유형
        - `l1` : L1 규제를 통해 과적합을 방지함
            - 맨해튼 거리 측정법에 기초한 오차 계산법을 통해 가중치를 규제함
            - 즉, 손실 함수에 가중치 절대값의 합을 더함
            - 특정 설명변수의 가중치를 0으로 만들 수 있음
            - feature selection
        - `l2` : L2 규제를 통해 과적합을 방지함
            - 유클리디안 거리 측정법에 기초한 오차 계산법을 통해 가중치를 규제함
            - 즉, 손실 함수에 가중치 제곱의 합을 더함
            - 이상치에 해당하는 가중치의 영향력을 최소화할 수 있음
    
    - `C = 1` : 가중치 규제 강도로서 손실 함수로 측정된 손실 크기
    
    - `max_iter = 100` : 경사 확인 횟수

- **다음의 속성을 통해 훈련된 모델의 정보를 확인할 수 있음**
    - `n_features_in_` : 설명변수의 수
    - `feature_nmaes_in_` : 설명변수명
    - `coef_` : 각 설명변수의 가중치
    - `intercept_` : 편향성

</details>

---

## 📝 Practice

- [**실습 코드**]()

- [**데이터 명세서**]()