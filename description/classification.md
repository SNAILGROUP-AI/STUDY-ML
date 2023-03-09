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
    - **이항분류분석(Binary Classification Analysis)** : 대상을 둘 중 하나로 분류하는 분석 방법
    - **다항분류분석(Multi-category Classificaiton Analysis)** : 대상을 3개 이상의 범주 중 하나로 분류하는 분석 방법

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
    
    - `max_depth = -1` : 트리 최대 깊이
    
    - `min_samples_split = 2` : 하위 노드로 가지치기하기 위해 필요한 최소한의 샘플 개수
    
    - `min_samples_leaf = 1` : 리프 노드가 되기 위해 필요한 최소한의 샘플 개수

- **다음의 속성을 통해 훈련된 모델의 정보를 확인할 수 있음**
    - `feature_importances_` : 설명변수별 가중치

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

- **기타 하이퍼파라미터 및 모델 정보 목록은 `estimator_params`을 통해 확인 가능함**

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

- **기타 하이퍼파라미터 및 모델 정보 목록은 `estimator_params`을 통해 확인 가능함**

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
    lg_clf = LogisticRegression()
    
    # 훈련용 데이터 세트를 통해 인스턴스를 훈련시켜서 모델 설계
    lg_clf.fit(X_train, y_train)

    # 평가용 데이터 세트를 통해 예측
    y_predict = lg_clf.predict(X_test)

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

- **기타 하이퍼파라미터 및 모델 정보 목록은 `estimator_params`을 통해 확인 가능함**

</details>

---

## 💯 성능 평가

<details><summary><h3>성능 평가란 무엇인가</h3></summary>

- **정의 : 실제 범주와 예측 범주의 일치 여부를 기준으로 모델의 성능을 평가하는 절차**

- **목적 : 과적합 방지 및 최적 모델 채택**
    - **과적합(overfitting) 판단 기준**
        - 학습용 데이터를 통해 수행한 정답률과 평가용 데이터를 통해 수행한 정답률 간 차이가 적을수록 과적합되지 않았다고 판단함
        - 학습용 데이터로는 성능이 높게 평가되었으나 평가용 데이터로는 성능이 낮게 평가되었다면 학습용 데이터에 과대적합된 상태임
        - 반대로 학습용 데이터로는 성능이 낮게 평가되었으나 평가용 데이터로는 성능이 높게 평가되었다면 학습용 데이터에 과소적합된 상태임

    - **최적 모델 판단 기준**
        - 과적합 문제가 해결된 여러 모델들 중에서 성능이 가장 높은 모델을 채택하는 기준으로 사용 가능함
        - 즉, 최적 알고리즘 및 최적 하이퍼파라미터 판단 기준으로서 사용 가능함

</details>

<details><summary><h3>오차행렬</h3></summary>

- **오차행렬(confusion matrix)**

    | | Predict Positive | Predict Negative |
    |---|---|---|
    | Actual Positive | True Positive | False Negative |
    | Actual Negative | False Positive | True Negative |

    - 정의 : 분류분석 결과 예측 범주와 실제 범주를 교차 표(cross table) 형태로 정리한 행렬
    - 용도 : 이항분류분석에 대하여 예측 오류가 얼마인지와 더불어 어떠한 유형의 예측 오류가 발생할 수 있는지 나타냄

- **해석**
    - **True Positive(TP)** : 예측값이 1이고, 실제값도 1인 경우
    - **True Negative(TN)** : 예측값이 0이고, 실제값도 0인 경우
    - **False Positive(FP)** : 제1종 오류; 예측값이 1이고, 실제값은 0인 경우
    - **False Nagative(FN)** : 제2종 오류; 예측값이 0이고, 실제값은 1인 경우

- **정확도, 정밀도, 재현율**

    | | Predict Positive | Predict Negative | |
    |---|---|---|---|
    | Actual Positive | True Positive | False Negative | Sensitivity |
    | Actual Negative | False Positive | True Negative | Specificity |
    | | Precision | | Accuracy |

    - **정확도** : $TP+TN \over TP+TN+FP+FN$

    - **정밀도** : $TP \over TP+FP$
    
    - **재현율** : $TP \over TP+FN$

</details>

<details><summary><h3>정확도(Accuracy)</h3></summary>

- **정의 : 전체 예측 개수 대비 정확하게 예측한 개수**
    - 해석 : 0~1 사이의 값을 가지며 1에 가까울수록 성능이 우수하다고 평가함
    - 목적 : 실제 데이터와 예측 데이터가 얼마나 동일한지를 평가 기준으로 하는 지표

- **주의 : 반응변수의 범주 간 개수가 불균형한 데이터 셋의 경우 활용하기에 적합하지 않음**
    - 가령 이항분류분석에서 참인 것의 개수가 99이고 거짓인 것의 개수가 1이라고 하자
    - 무조건 참으로 예측하면 0.99의 정확도를 가지게 됨

</details>

<details><summary><h3>F1-Score</h3></summary>

- **정의 : 정밀도와 재현율의 조화 평균**
    - 해석 : 0~1 사이의 값을 가지며 1에 가까울수록 성능이 우수하다고 평가함
    - 목적 : 정밀도와 재현율 중 어느 한쪽으로 치우치지 않을수록 높은 값을 가짐

- **정밀도**
    - 정의 : 참으로 예측한 것의 개수 대비 정확하게 예측한 개수
    - 해석 : 0~1 사이의 값을 가지며 1에 가까울수록 성능이 우수하다고 평가함
    - 용도 : 제1종 오류가 문제되는 경우 주요한 지표로서 사용됨
    - 즉, 실제 거짓인 데이터를 참으로 판단하면 큰 문제가 발생하는 경우

- **재현율**
    - 정의 : 참인 것의 개수 대비 참으로 예측한 것의 개수로서 민감도(sensitivity)라고도 부름
    - 해석 : 0~1 사이의 값을 가지며 1에 가까울수록 성능이 우수하다고 평가함
    - 용도 : 제2종 오류가 문제되는 경우 주요한 지표로서 사용됨
    - 즉, 실제 참일 데이터를 거짓으로 판단하면 큰 문제가 되는 경우

- **정밀도와 재현율의 관계**
    - **정밀도와 재현율은 모두 TP를 높이는 것을 목적으로 함**
        - 단, 정밀도는 제1종 오류에 초점을 맞추는 지표로서 FP를 낮추는 방향으로 TP를 높이고자 함
        - 반면, 재현율은 제2종 오류에 초점을 맞추는 지표로서 FN를 낮추는 방향으로 TP를 높이고자 함

    - **정밀도와 재현율은 Trade-off 관계라고 볼 수 있음**
        - FP와 FN 중 어느 한쪽의 수치를 강제로 높이면 다른 한쪽의 수치를 낮추기 쉬워짐
        - 가령 어떤 자료가 참일 확률이 0.9라면 참으로 에측할 가능성이 매우 높음
        - 반면, 어떤 자료가 참일 확률이 0.1이라면 거짓으로 예측할 가능성이 매우 높음

    - **정밀도와 재현율의 조화평균이 가지는 의미**
        - 그렇다면 어떤 자료가 참일 확률이 0.6이라면 참과 거짓 중 무엇으로 분류해야 하는가
        - 임계값(threshold)을 기준으로 분류할 수 있음
        - F1-Score은 정확도와 재현율 중 어느 한쪽을 희생하지 않고서 양쪽을 모두 높이는 임계치임

</details>

<details><summary><h3>AUC</h3></summary>

- **ROC 곡선**
    - **민감도(True Positive Rate; TPR)** :  참인 것에 대하여 참으로 예측한 비율
    - **특이도(True Negative Rate; TNR)** : 거짓인 것에 대하여 거짓으로 예측한 비율
    - **ROC 곡선** : $1-TNR$ 의 변화에 따른 TPR의 변화 양상을 나타내는 곡선
    - $(x, y)=(0, 1)$ 일 때 성능이 가장 좋음

- **민감도와 특이도는 trade-off 관계임**
    - 모든 자료를 참으로 예측하는 경우 민감도를 최대치로 가져갈 수 있음
    - 반면, 이러한 경우 특이도를 최소치로 가져가게 됨

- **AUC(Area Under Curve)**
    - 정의 : ROC 곡선과 X축으로 둘러싸인 면적의 너비
    
    - 해석 : 0.5~1의 값을 가지며 1에 가까울수록 성능이 우수하다고 평가함
    
    - 장점
        - **척도 불변(Scale-Invariant)** : 절대값이 아니라 비율을 통해서 성능을 평가함
        - **분류 임계값 불변(Classification-Threshold-Invariant)** : 어떤 분류 임계값으로 무엇을 선택했는지와 무관하게 성능을 평가함

</details>

---

## 📝 Practice

- [**실습 코드**]()

- [**데이터 명세서**]()