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

## 💯 성능 평가

<details><summary><h3>성능 평가란 무엇인가</h3></summary>

- **정의 : 오차를 기준으로 모델의 성능을 평가하는 절차**
    - 오차 : 예측값과 실제값의 차이
    - 오차를 0으로 만드는 것은 현실적으로 불가능하므로 오차를 허용할 범위를 결정해야 함
    - 실제값이 주어져야 오차를 측정할 수 있으므로 지도학습에서만 이루어짐

- **목적 : 과적합 방지 및 최적 모델 채택**
    - **과적합(overfitting) 판단 기준**
        - 학습용 데이터를 통해 수행한 예측 오차와 평가용 데이터를 통해 수행한 예측 오차 간 차이가 적을수록 과적합되지 않았다고 판단함
        - 학습용 데이터로는 성능이 높게 평가되었으나 평가용 데이터로는 성능이 낮게 평가되었다면 학습용 데이터에 과대적합된 상태임
        - 반대로 학습용 데이터로는 성능이 낮게 평가되었으나 평가용 데이터로는 성능이 높게 평가되었다면 학습용 데이터에 과소적합된 상태임

    - **최적 모델 판단 기준**
        - 과적합 문제가 해결된 여러 모델들 중에서 성능이 가장 높은 모델을 채택하는 기준으로 사용 가능함
        - 즉, 최적 알고리즘 및 최적 하이퍼파라미터 판단 기준으로서 사용 가능함

</details>

<details><summary><h3>편차와 오차의 구분</h3></summary>

- **모집단과 표본집단**
    - **모집단(Poplulation)** : 통계 조사의 대상이 되는 집단 전체
    - **표본(Sample)** : 모집단에서 어떠한 방법으로 선발된 일부 원소들의 집합으로서 모집단의 부분집합
    - **표본 집단** : 표본들의 집합

- **편차와 표준편차**
    - **편차(Deviation)** : 개별값과 대표값(==평균)의 차이
    - **표준편차(Standard Deviation)** : 편차들의 대표값(==평균)
    - **목적** : 대표값이 자료를 얼마나 잘 대표하고 있는가

- **오차와 표준오차**
    - **오차(Error)** : 특정 표본의 통계량과 모수의 차이
        - **모수(Parameter)** : 모집단을 묘사하는 측정치로서 모평균, 모분산, 모표준편차 등이 있음
        - **통계량(Statistic)** : 모수에 대한 추정치 혹은 표본을 묘사하는 측정치
    
    - **표준오차(Standard Error)** : 오차들의 대표값

    - **목적** : 표본이 모집단을 얼마나 잘 추론하고 있는가

- **편차와 오차의 구분**
    - **편차는 기술통계학의 측정치에 해당함**
        - **기술통계학(Descriptive Statistcs)** : 자료를 수집, 정리, 제시, 요약함
        - **측정치(Measure)** : 자료의 형태를 묘사하거나 요약하는 수치
    
    - **오차는 추론통계학의 추정치에 해당함**
        - **추론통계학(Inferential Statistic)** : 표본으로부터 모집단의 성격을 추론함
        - **추정치(Estimate)** : 모집단의 측정치를 추론하거나 그 정확성을 추론하는 수치

</details>

<details><summary><h3>결정계수</h3></summary>

### $$\displaystyle\sum_{i=1}^{n}{\frac{(y_i-\hat{y})^2}{(y_i-\overline{y})^2}}$$

- **결정계수(Coefficient of Determination; r2-score)**

    - 정의 : 모분산 대비 표본분산 비율
    - 해석 : 0~1 사이의 값을 가지며, 값이 클수록 회귀식의 적합도가 높다고 판단함

</details>

<details><summary><h3>평균제곱오차</h3></summary>

### $$MSE=\displaystyle\sum_{i=1}^{n}{\frac{(y_i-\hat{y})^2}{n}}$$

- **평균제곱오차(Mean Squared Error; MSE)**

    - 정의 : 오차를 제곱한 값의 평균
    - 해석 : 값이 작을수록 회귀식의 적합도가 높다고 판단함
    - 문제점 : 오차를 제곱하므로 값을 과장할 수 있음

</details>

<details><summary><h3>평균제곱근오차</h3></summary>

### $$RMSE=\sqrt{\displaystyle\sum_{i=1}^{n}{\frac{(y_i-\hat{y})^2}{n}}}$$

- **평균제곱근오차(Root Mean Squared Error; RMSE)**

    - 정의 : 평균제곱오차의 제곱근
    - 목적 : 평균제곱오차에 제곱근하는 절차를 더하여 오차의 크기가 과장된 정도를 줄임

</details>

<details><summary><h3>평균절대오차</h3></summary>

### $$RAE=\displaystyle\sum_{i=1}^{n}{\frac{|y_i-\hat{y_i}|}{n}}$$

- **평균절대오차(Mean Absolute Error; MAE)**

    - 정의 : 오차 절대값의 평균
    - 목적 : 오차를 제곱한 값 대신 오차의 절대값을 활용하여 오차의 크기가 과장될 여지를 없앰

</details>

---

## 📝 Practice

- [**실습 코드**]()

- [**데이터 명세서**]()