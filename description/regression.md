## 🤓 회귀분석이란 무엇인가

- **정의 : 데이터를 요약하는 수렴 작업**
    - 데이터의 추세를 설명하는 한 문장을 찾는 작업
    - 반응변수가 수치형 변수인 경우

- **학습 양상**
    - **학습** : 최적 회귀계수 조합을 찾는 행위
    - **최적 회귀계수** : 잔차항의 크기를 최소화하는 회귀계수
    - **잔차항** : 실제값과 예측값의 차이
    - **회귀계수** : 가중치(w)와 편향성(b)

- **구분**
    - **설명변수의 개수에 따른 구분**
        - **단순회귀분석(Simple egression Analysis)** : 반응변수에 대한 설명변수가 하나인 경우
        - **다중회귀분석(Multiple Regression Analysis)** : 반응변수에 대한 설명변수가 두 가지 이상인 경우
    
    - **선형 가정 여부에 따른 구분**
        - **선형회귀분석** : 설명변수와 반응변수 간 선형관계를 가정하는 회귀분석
            - **오차(혹은 잔차)의 크기를 최소화하는 방법에 따라 다시 두 가지로 구분할 수 있음**
                - 최소제곱법(Ordinary Least Squares; OLS)
                - 경사하강법(Stochastic Gradient Descent; SGD)
            
            - **여기서는 선형회귀분석 알고리즘만을 서술하겠음**
                - 선형 회귀 알고리즘 : 최고제곱법에 기반한 선형회귀분석 알고리즘
                - 확률적 경사 하강 회귀 알고리즘 : 경사하강법에 기반한 선형회귀분석 알고리즘
        
        - **비선형회귀분석** : 설명변수와 반응변수 간 선형관계를 가정하지 않는 회귀분석
            - **대표적인 비선형회귀분석 알고리즘으로는 회귀 트리 알고리즘이 있음**
                - 전반적인 작동방식은 분류분석의 결정 트리 알고리즘과 유사함
                - 단, 리프 노드에서 범주를 결정하지 않고 평균값을 구하여 추세를 예측함
                - `sklearn.tree`의 `DecisionTreeRegressor` 등
            
            - **트리 기반 API의 경우 비선형회귀분석 알고리즘을 지원함**
                - `RandomForestRegressor`, `GradientBoostingRegressor` 등

- **선형회귀분석의 수학적 이해**
    - 수치형 반응변수 $Y$ 가 설명변수 $X$ 와 선형 관계에 있다고 가정하자

    - 샘플 $i$ 에 대하여 반응변수와 설명변수의 회귀식은 다음과 같음
        
        ### $$Y_i=b+wX_i+e_i$$

        - $b$ : 편향성; 설명변수의 영향력이 모두 제거되었을 때 반응변수의 상태
        - $w$ : 가중치; 반응변수 $Y_i$ 에 대한 설명변수 $X_i$ 의 영향력
        - $e_i$ : 잔차항; 편향성과 가중치의 조합만으로는 설명될 수 없는 항목의 모음

    - 잔차항 혹은 오차는 실제값과 예측값의 차이로서 구체적으로 다음을 의미함

        ![image](https://user-images.githubusercontent.com/116495744/221339174-de431950-85c5-4156-afbc-0d3ba0b9c8e4.png)

---

## 📈 선형 회귀(Linear Regression)

<details><summary><h3>선형 회귀란 무엇인가</h3></summary>

- **정의 : 최소제곱법을 통해 회귀식을 도출하는 알고리즘**

- **최소제곱법(Ordinary Least Squares; OLS)**
    - 정의 : 잔차 제곱의 합을 최소화하는 회귀식을 도출하는 방법
    - 잔차 제곱의 합을 최소화한다는 것은 다음을 의미함
        
        ### $$\min\displaystyle\sum_{i=1}^ne_i^2=\min\sum_{i=1}^n(Y_i-b-wX_i)^2$$

</details>

<details><summary><h3>SK-Learn의 선형 회귀 알고리즘</h3></summary>
    
- **사용 방법**

    ```
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    # 선형 회귀 알고리즘 인스턴스 생성
    li_reg = LinearRegression()

    # 훈련용 데이터 세트를 통해 인스턴스를 훈련시켜서 모델 설계
    li_reg.fit(X_train, y_train)

    # 평가용 데이터 세트를 통해 예측
    y_predict = li_reg.predict(X_test)
    
    # 대표적인 성능 평가 지표인 결정계수를 통해 성능 평가
    score = r2_score(y_test, y_predict)
    print(score)
    ```

- **다음의 속성을 통해 훈련된 모델의 정보를 확인할 수 있음**
    - `n_features_in_` : 설명변수의 수
    - `feature_nmaes_in_` : 설명변수명
    - `coef_` : 각 설명변수의 가중치
    - `intercept_` : 편향성

- **기타 하이퍼파라미터 및 모델 정보 목록은 `estimator_params`을 통해 확인 가능함**

</details>

---

## 📉 확률적 경사 하강 회귀(Stochastic Gradient Descent Regression; SGD)

<details><summary><h3>확률적 경사 하강 회귀란 무엇인가</h3></summary>

- **정의 : 경사하강법을 통해 회귀식을 도출하는 알고리즘**

- **확률적 경사하강법(Stochastic Gradient Descent; SGD)**
    - **정의 : 최적화된 손실함수에 근거하여 회귀식을 도출하는 방법**
    
    - **손실함수(Loss Function)**
        - 정의 : 손실을 반응변수, 가중치의 조합을 설명변수로 가지는 함수
            - 평균제곱오차(MSE)에 기초한 손실함수는 다음으로 정의됨

                ### $$LOSS_{MSE}=\frac{1}{N}\displaystyle\sum_{i=1}^n (\hat{Y_i}-Y_i)^2$$

        - **손실(Loss)** : 어떠한 방법에 따라 잔차를 계산한 값

        - **최적화(Optimizing)** : 손실을 최소화하는 가중치 조합을 찾는 일

    - **왜 경사를 하강하는 방법이라고 부르는가?**
        - 손실함수의 반응변수인 손실을 최소화하는 일계조건은 그 도함수의 반응변수가 0을 만족하는 것임
        - 도함수의 반응변수는 원함수의 경사(Gradient)를 나타냄
        - 따라서 확률적 경사하강법은 원함수의 경사가 수평이 되는 지점을 찾는 일이라고 볼 수 있음

- **주요 이슈 : 학습률(Learning Rate)**
    - **정의 : STEP의 단위 혹은 보폭**
    - 확률적 경사 하강 회귀 알고리즘은 다음의 절차를 통해 경사가 0에 근사한 지점을 찾음
    - 즉, 임의로 선택된 손실함수 그래프의 한 점에서 시작하여 학습률만큼 움직이면서 경사를 확인함
    - 학습률은 정확도에 비례하고, 처리 속도에 반비례함

</details>

<details><summary><h3>SK-Learn의 확률적 경사 하강 회귀 알고리즘</h3></summary>

- **사용 방법**

    ```
    from sklearn.linear_model import SGDRegressor
    from sklearn.metrics import r2_score

    # 확률적 경사 하강 회귀 알고리즘 인스턴스 생성
    sgd_reg = SGDRegressor()

    # 훈련용 데이터 세트를 통해 인스턴스를 훈련시켜서 모델 설계
    sgd_reg.fit(X_train, y_train)

    # 평가용 데이터 세트를 통해 예측
    y_predict = sgd_reg.predict(X_test)
    
    # 대표적인 성능 평가 지표인 결정계수를 통해 성능 평가
    score = r2_score(y_test, y_predict)
    print(score)
    ```

- **주요 하이퍼파라미터**
    - `learning_rate = 0.1` : 학습률

- **다음의 속성을 통해 훈련된 모델의 정보를 확인할 수 있음**
    - `n_features_in_` : 설명변수의 수
    - `feature_nmaes_in_` : 설명변수명
    - `coef_` : 각 설명변수의 가중치
    - `intercept_` : 편향성

- **기타 하이퍼파라미터 및 모델 정보 목록은 `estimator_params`을 통해 확인 가능함**

</details>

---

## 📝 Practice

- [**실습 코드**]()

- [**데이터 명세서**]()