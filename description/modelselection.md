<h1 align="center"> ✅ Model Selection ✅ </h1>

---

## ✂︎ 데이터 세트 나누기

<details><summary><h3>train_test_split</h3></summary>

![데이터 세트 나누기](https://miro.medium.com/max/1400/0*DKB-pJy7-G6gEkM-)

- **목적 : 인스턴스 훈련에 사용할 데이터 세트와 성능 평가에 사용할 데이터 세트를 구분하기 위함**
    
- **용도**
    - **훈련용(train)** : 인스턴스 훈련(혹은 학습) 용도
    - **검증용(validation)** : 과적합을 충분히 방지하여 훈련을 중단해도 무방한지 판단 용도
    - **평가용(test)** : 인스턴스 성능 평가 용도로서 훈련 시 사용되지 않은 레코드

- **사용 방법**

    ```
    from sklearn.model_selection import train_test_split

    # 데이터 세트를 설명변수 조합 X와 반응변수 y로 구분함
    X = df.drop(columns = [target])
    y = df[[target]]

    # 데이터 세트를 훈련용과 평가용으로 분리함
    # 훈련용 데이터 세트를 (X_train, y_train), 평가용 데이터 세트를 (X_test, y_test)에 할당함
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    ```

- **주요 하이퍼파라미터**
    - `random_state = None`

    - `test_size = 0.25` : 전체 데이터 세트 대비 평가용 데이터 세트의 비중
    
    - `shuffle = True` : 훈련용과 평가용으로 분리하기 전에 레코드를 무작위로 섞을 것인가
    
    - `stratify = None` : 범주의 비율을 훈련용과 평가용에도 유지할 범주형 변수 목록
        - 분류분석 시 반응변수에 대하여 설정할 것을 권장함

</details>

---

## 🫵 과적합

<details><summary><h3>Overfitting & Underfitting</h3></summary>

![](https://gratus907.github.io/images/81b7294441f2b9c96cce938661b95a1d20d22366e5c0f72e48d2c69c9c7ad7b4.png)

- **정의 : 모델이 일반화되지 못하는 현상**

- **종류**
    - **과대적합(OverFitting)**
        - 정의 : 모델이 학습용 데이터 세트에 과도하게 최적화된 상태
        - 주요 원인 : 모델 복잡도 심화
        - 현상 : 학습용 데이터 세트 예측 성능과 그 이외의 데이터 세트 예측 성능 간 격차가 상당함

    - **과소적합(UnderFitting)**
        - 정의 : 알고리즘이 학습용 데이터 세트의 규칙을 제대로 찾지 못하여 모델이 단순하게 설계된 상태 
        - 주요 원인 : 학습할 데이터 수 부족
        - 현상 :  학습용 데이터 세트 예측 성능과 그 이외의 데이터 세트 예측 성능 모두 현저하게 낮음

- **해법 : 교차검증**
    - **k-겹 교차검증** : 
    - **층화 k-겹 교차검증** : 반응변수의 범주 비율이 모집단의 비율과 동일하도록 구성하여 교차검증함

</details>

<details><summary><h3>K-Fold</h3></summary>

![k fold 교차검증](https://i0.wp.com/drzinph.com/wp-content/uploads/2020/12/image-2.png?fit=935%2C670&ssl=1)

</details>

<details><summary><h3>Stratified K-Fold</h3></summary>

![stratified k fold 교차검증](https://i0.wp.com/dataaspirant.com/wp-content/uploads/2020/12/8-Stratified-K-Fold-Cross-Validation.png?ssl=1)

</details>

<details><summary><h3>cross_val_score</h3></summary>

</details>

---

## ✍️ 하이퍼파라미터 튜닝

<details><summary><h3>GridSearchCV</h3></summary>

- **사용 방법**

    ```
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier

    # 하이퍼파라미터를 튜닝하려는 알고리즘 인스턴스 생성
    model = KNeighborsClassifier()

    # 하이퍼파라미터와 해당 파라미터의 아규먼트 목록 생성
    # 하이퍼파라미터명을 key, 아규먼트 목록을 value로 가지는 dictionary type
    params = {
        "metric" : ["minkowsi", "euclidean", "manhattan"],
        "n_jobs" : [None, -1],
        "n_neighbors" : range(1, 10),
        "weights" : ["uniform", "distance"]
    }

    # 교차검증 횟수를 3회로 설정
    cvNum = 3

    # GridSearchCV 인스턴스 생성
    # 성능이 가장 높게 검증된 하이퍼파라미터 조합에 대하여 재학습하여 재검증함
    gridModel = GridSearchCV(
        model,
        param_grid = params,
        cv = cvNum,
        refit = True
    )

    # 학습용 데이터 세트를 통해 최적 하이퍼파라미터 탐색
    gridModel.fit(X_train, y_train)

    # 탐색 결과 비교
    score_df = pd.DataFrame(gridModel.cv_results_)

    param_col = [f"param_{i}" for i in params]
    score_col = ["rank_test_score", "mean_test_score", "std_test_score"]
    cv_col = [f"split{i}_test_score" for i in range(cvNum)]
    col_list = param_col + score_col + cv_col

    score_df = score_df[col_list]

    print(score_df)

    # 최적하이퍼파라미터 확인
    print(gridModel.best_params_)
    ```

- **주요 하이퍼파라미터**

</details>