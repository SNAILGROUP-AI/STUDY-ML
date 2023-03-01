## 👨‍👩‍👦 Ensemble

- **앙상블 기법이란 무엇인가**
    - 단일 모델들을 결합하는 기법
    - 회귀 알고리즘에도 앙상블 기법을 적용할 수 있으나, 주로 분류 알고리즘에 적용함

- **앙상블 기법의 종류**
    - **Voting** : 여러 분류 모델들을 수동으로 결합하는 방식
        - **Hard Voting** : 임의의 자료에 대하여 모델들이 예측한 결과를 종합하여 다수결로 해당 자료의 범주를 선정하는 방식
        - **Soft Voting** : 임의의 자료에 대하여 모델들이 예측한 범주별 확률의 평균을 낸 후 확률이 가장 높은 범주를 해당 자료의 범주로 선정하는 방식
    
    - **Bagging(Bootstrap Aggregating)** : 여러 분류 모델들을 병렬로 작업하도록 설계하는 방식
    
    - **Boosting** : 여러 분류 모델들을 직렬로 심화 작업하도록 설계하는 방식
        - 첫 번째 알고리즘을 통해 훈련함
        - 손실(혹은 오차)이 높은 자료들에 가중치를 부여하여 두 번째 알고리즘으로 전송함
        - 두 번째 알고리즘을 통해 훈련하되, 손실이 높은 자료를 심화 훈련함
        - 이상의 작업을 반복함
        - `learning_rate`(학습률)을 통해 얼마나 가중할 것인지 설정함

---

## ✅ Voting

<details><summary><h3>Example</h3></summary>

- **사용 방법**

    ```
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import accuracy_score

    # 결정 트리 알고리즘 인스턴스 생성
    dt_clf = DecisionTreeClassifier()

    # 최근접 이웃 알고리즘 인스턴스 생성
    knn_clf = KNeighborsClassifier()

    # 로지스틱 회귀 알고리즘 인스턴스 생성
    lg_clf = LogisticRegression()

    # soft voting을 통해 세 인스턴스를 결합하여 앙상블 알고리즘 인스턴스 생성
    voting_clf = VotingClassifier(
        estimators = [('DT', dt_clf), ('KNN', knn_clf), ('LG', lg_clf)],
        voting = 'soft'
        )

    # 훈련용 데이터 세트를 통해 인스턴스를 훈련시켜서 모델 설계
    voting_clf.fit(X_train, y_train)

    # 평가용 데이터 세트를 통해 예측
    y_predict = voting_clf.predict(X_test)

    # 대표적인 성능 평가 지표인 결정계수를 통해 성능 평가
    score = accuracy_score(y_test, y_predict)
    print(score)
    ```

- **주요 하이퍼파라미터**
    - `estimators` : 동원할 알고리즘 인스턴스 리스트
    
    - `voting = hard` : 설계 방식
        - `hard` : hard voting
        - `soft` : soft voting

</details>

---

## 🤝 Bagging

<details><summary><h3>Random Forest</h3></summary>

- **정의**

- **사용 방법**

    ```

    ```

- **주요 하이퍼파라미터**

</details>

---

## 🕵️ Boosting

<details><summary><h3>Gradient Boosting Machine(GBM)</h3></summary>

- **정의**

- **사용 방법**

    ```
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    
    # GBM 알고리즘 인스턴스 생성
    gb_clf = GradientBoostingClassifier(
        random_state = 121, 
        learning_rate = 0.1, 
        n_estimators = 100)

    # 훈련용 데이터 세트를 통해 인스턴스를 훈련시켜서 모델 설계
    gb_clf.fit(X_train, y_train)
    
    # 평가용 데이터 세트를 통해 예측
    y_predict = gb_clf.predict(X_test)
    
    # 대표적인 성능 평가 지표인 결정계수를 통해 성능 평가
    socre = accuracy_score(y_test, y_predict)
    print(score)
    ```

- **주요 하이퍼파라미터**

</details>

<details><summary><h3>eXtra Gradient Boosting(XGBoost)</h3></summary>

- **정의**
    - 병렬 처리가 불가능하여 속도가 느리고 과적합이 발생할 우려가 있는 GBM 알고리즘을 보완함
    - 내장된 교차검증 절차를 통해 예측 성능이 향상되지 않을 경우 Early Stopping할 수 있음

- **사용 방법**

    ```
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    
    # XGB 알고리즘 인스턴스 생성
    xgb_clf = XGBClassifier(
        random_state = 121, 
        n_estimators = 100, 
        learning_rate = 0.1
        )
    
    # 교차검증 시 사용할 데이터 세트 구성
    evals = [(X_val, y_val)]

    # 훈련용 데이터 세트를 통해 인스턴스를 훈련시켜서 모델 설계
    # 검증용 데이터 세트를 통해 교차검증
    # 훈련 최대 횟수를 400회로 제한
    xgb_clf.fit(
        X_train, y_train,
        early_stopping_rounds = 400, 
        eval_set = evals,
        eval_metric = 'logloss'
        )

    # 평가용 데이터 세트를 통해 예측
    y_predict = xgb_clf.predict(X_test)
    
    # 대표적인 성능 평가 지표인 결정계수를 통해 성능 평가
    socre = accuracy_score(y_test, y_predict)
    print(score)
    ```

- **주요 하이퍼파라미터**
    - `random_state = None`

    - `n_estimators = 100` : 동원할 모델의 개수
    
    - `learning_rate = 0.1` : 학습률
    
    - `early_stopping_rounds` : 학습이 장기화될 경우 조기 종료하기 위하여 최대 학습 횟수 설정
    
    - `eval_set` : 성능 교차검증에 사용할 데이터 세트
        - 데이터 세트 분할 시 우선 훈련용과 평가용으로 분할함
        - 이후 훈련용 데이터 세트를 훈련용과 검증용으로 재분할함
    
    - `eval_metric` : 교차검증 시 사용할 평가 지표
        - `logloss` : 이항분류분석 교차검증 시 평가 지표
        - `multi-logloss` : 다항분류분석 교차검증 시 평가 지표

- **다음을 통해 학습된 모델이 계산한 설명변수별 가중치를 확인할 수 있음**

    ```
    from xgboost import plot_importance
    print(plot_importance(xgb_clf))
    ```

</details>

<details><summary><h3>Light Gradient Boosting Machine(LightGBM)</h3></summary>

- **정의**

- **사용 방법**

    ```
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    
    gb_clf = GradientBoostingClassifier(
        random_state = 121, 
        learning_rate = 0.1, 
        n_estimators = 100)

    gb_clf.fit(X_train, y_train)
    
    y_predict = gb_clf.predict(X_test)
    
    socre = accuracy_score(y_test, y_predict)
    print(score)
    ```

- **주요 하이퍼파라미터**

</details>