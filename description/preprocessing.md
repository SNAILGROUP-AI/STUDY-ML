## ✂︎ 데이터 세트 나누기

<details><summary><h3>데이터 세트 나누기</h3></summary>

![](https://miro.medium.com/max/1400/0*DKB-pJy7-G6gEkM-)

- **목적 : 인스턴스 훈련에 사용할 데이터 세트와 성능 평가에 사용할 데이터 세트를 구분하기 위함**
    
- **용도**
    - **훈련용(train)** : 인스턴스 훈련(혹은 학습) 용도
    - **검증용(validation)** : 과적합을 충분히 방지하여 훈련을 중단해도 무방한지 판단 용도
    - **평가용(test)** : 인스턴스 성능 평가 용도로서 훈련 시 사용되지 않은 레코드

- **사용 방법**

    ```
    from sklearn.model_selection import train_test_split

    # 데이터 세트를 설명변수 조합 X와 반응변수 y로 구분함
    # 반응변수가 데이터프레임의 마지막 컬럼이라고 가정함
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

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

## ❓ 결측치 처리

<details><summary><h3>결측치(Missing Value)란 무엇인가</h3></summary>

- **정의 : 기입되지 않은 데이터**

- **`None` 과 `NaN` 의 구분**
    - `None`(Null) : 아무것도 존재하지 않는 데이터
    - `NaN`(Not a Number) : 정의되거나 표현되지 못하는 데이터

- **발생 원인 규명 가능 여부에 따른 구분**
    - **완전 무작위 결측(Missing completely at Random; MCA)** : 발생 원인을 파악할 수 없는 결측치
    - **무작위 결측(Missing at Random; MAR)** : 발생 원인을 완전히 설명할 수는 없는 결측치
    - **비무작위 결측(Missing at Not Random; MNAR)** : 발생 원인을 완전히 설명할 수 있는 결측치

- **결측치 처리 권장 방식**
    - **비무작위 결측의 처리**
        - 대표값에 영향을 미치지 않는 특정 값으로 대체함
            - 결측치를 나타내는 특정 값을 정의하고 해당 값으로 대체
            - 평균(수치형 변수), 최빈값(범주형 변수) 등 대표값으로 대체
    
    - **무작위 결측의 처리**
        - 10% 미만 : 해당 레코드 제거
        - 20% 미만 : 평균(수치형 변수), 최빈값(범주형 변수) 등 대표값으로 대체
        - 20% 이상 : 결측치를 처리하는 머신러닝 모델 설계

</details>

<details><summary><h3>결측치 탐색</h3></summary>
    
- **메소드 `isnull()`를 통한 결측치 탐색**

    ```
    # 결측치가 존재할 경우 True, 존재하지 않을 경우 False를 기입한 데이터프레임 반환
    df.isnull()

    # 데이터프레임 df의 컬럼별 결측치 개수 반환
    df.isnull().sum()

    # 데이터프레임 df의 컬럼별 결측치 비율 반환
    df.isnull().mean()
    ```

- **모듈 `missingno`를 통한 결측치 분포 시각화**

    ```
    import missingno as msno

    # 데이터프레임 df의 컬럼별 결측치 위치 시각화
    msno.matrix(df = df)
    
    # 데이터프레임 df의 컬럼별 결측치 비율 시각화
    msno.bar(df = df)
    ```

</details>

<details><summary><h3>결측치 처리</h3></summary>

- **메소드 `dropna()`를 통한 결측치가 포함된 레코드 제거**

    ```
    df.dropna()
    ```

    - `how = 'any'` : 삭제 조건 세부 설정
        
        - `any` : 결측치가 하나라도 포함된 레코드를 제거함
        - `all` : 모든 컬럼이 결측치인 레코드만 제거함

- **메소드 `fillna()`를 통한 결측치 대체**

    ```
    # 범주형 설명변수의 결측치를 최빈값으로 대체함
    mode_value = df[cat_col].mode(
        axis = 1, 
        numeric_only = False, 
        dropna = True
        )
    df[cat_col] = df[cat_col].fillna(mode_value)

    # 수치형 설명변수의 결측치를 평균으로 대체함
    mean_value = df[num_col].mean()
    df[num_col] = df[num_col].fillna(mean_value)
    ```

</details>

---

## 🔢 수치형 설명변수의 전처리

<details><summary><h3>이상치가 존재하는 경우</h3></summary>

- **이상치(Outlier)**
    - **정의 : 관측된 데이터의 범위에서 지나치게 벗어나 값이 매우 크거나 작은 값**

    - **이상치의 판별**
        - 제1사분위수와 제3사분위수가 상식과 부합하지 않는다면 데이터 세트가 잘못된 것으로 판단함
        - boxplot 등 분포 시각화 툴을 활용하여 이상치 존재 가능성 여부를 확인함
        - 이상치가 존재할 가능성이 있다고 판단되면 이상치 탐지 기법을 통해 이상치를 규정하고 처리함

    - **이상치의 탐지 : Turkey Fence 기법**
        - 정의 : 사분위 범위(InterQuartile Range; IQR)을 활용하여 이상치를 판별하는 기법
            - **사분위 범위(IQR)** : 제3사분위수(Q3) - 제1사분위수(Q1)

        - 이상치를 상한값을 초과하거나 하한값에 미달한 값으로 규정함
            - **하한값(lower_value)** : $Q1-IQR \times 1.5$
            - **상한값(upper_value)** : $Q3+IQR \times 1.5$

    - **이상치의 처리 : 통상적으로는 상한값 및 하한값으로 대체함**

- **사용 방법**

    ```
    from sklearn.preprocessing import RobustScaler
    
    # Turkey Fence 기법에 기반한 이상치 탐지 및 처리기 RobustScaler 인스턴스 생성
    scaler = RobustScaler()

    # 이상치 탐지
    scaler.fit(X_train)

    # 이상치 처리
    X_train = scaler.transform(X_train)
    ```

- **다음을 통해 스케일러의 정보를 확인할 수 있음**
    - `center_` : 중앙값
    - `scale_` : 사분위 범위

</details>

<details><summary><h3>분포가 들쑥날쑥한 경우</h3></summary>

- **표준화(Standardization)**

    ![stanard](https://user-images.githubusercontent.com/116495744/222760130-bdcce494-0d8b-407c-8859-6ab6524b6127.jpg)

    ### $$x_{new}=\frac{x_i-mean(x)}{std(x)}$$

    - 정의 : 값의 분포를 평균이 0, 분산이 1인 표준정규분포(가우시안 정규 분포) 형태로 변환함
    - 목적 : 모든 설명변수의 형태를 통계 분석의 가정에 부합하는 형태로 변환함

- **사용 방법**

    ```
    from sklearn.preprocessing import StandardScaler
    
    # 표준화 처리기 StandardScaler 인스턴스 생성
    scaler = StandardScaler()

    # 평균 및 분산 탐색
    scaler.fit(X_train)

    # 표준화
    X_train = scaler.transform(X_train)
    ```

</details>

<details><summary><h3>단위가 들쑥날쑥한 경우</h3></summary>

- **정규화(Normalization)**

    ![minmax](https://user-images.githubusercontent.com/116495744/222760155-d4fc55ff-3959-4b12-9acb-577c632ad958.jpg)

    ### $$x_{new}=\frac{x_i-min(x)}{max(x)-min(x)}$$

    - 정의 : 값의 범위를 특정하고 모든 설명변수의 분포를 해당 범위로 확대 혹은 축소함
    - 목적 : 모든 설명변수의 크기를 통일하여 설명변수 간 상대적 크기가 주는 영향력을 최소화함

- **사용 방법**

    ```
    from sklearn.preprocessing import MinMaxScaler
    
    # 정규화 처리기 MinMaxScaler 인스턴스 생성
    scaler = MinMaxScaler()

    # 최대최소 변환을 위한 분포 탐색
    scaler.fit(X_train)

    # 정규화
    X_train = scaler.transform(X_train)
    ```

</details>

<details><summary><h3>수치형 설명변수의 전처리 순서</h3></summary>

![스케일링 비교](https://miro.medium.com/max/1400/1*0Ox-p57oxfmaVSaJyJWyPg.png)

- **`RobustScaler` 👉  `StandardScaler` 👉 `MinMaxScaler` 순을 권장함**

    - 이상치가 존재할 경우 정규화에 따른 성능 개선 효과가 미미함
    - 정규화 이후 표준화를 하는 경우 설명변수별 범위가 재조정될 가능성이 있음

</details>

---

## 🔤 범주형 설명변수의 전처리

<details><summary><h3>인코딩</h3></summary>

</details>

---

## 📊 분류분석

<details><summary><h3>무의미한 설명변수 제거</h3></summary>

- **승산(odds)**
    - 이항범주형 반응변수에 대하여 반응하지 않을 가능성($1-p$) 대비 반응할 가능성($p$)
    - 반응변수가 반응할 가능성이 반응하지 않을 가능성보다 몇 배 높은가
    - 반응변수가 반응할 가능성을 $p$ 라고 했을 때, 승산 $odds$ 는 다음과 같음
    
    ### $$odds=\frac{p}{1-p}$$

- **승산비(Oods Ratio; OR)**
    - 이항범주형 반응변수 y와 이항범주형 설명변수 x에 대하여 x의 변동에 따른 y의 반응
    - 설명변수가 참일 때 반응변수가 반응할 가능성이 거짓일 때보다 몇 배 높은가
    - 설명변수 x가 참일 가능성을 $q$, 반응변수가 반응할 가능성을 $p$ 라고 했을 때, 승산비 $OR$ 은 다음과 같음

    ### $$OR=\frac{q \times \frac{p}{1-p}}{(1-q) \times \frac{p}{1-p}}$$

- **사용 방법**

    ```
    ```

</details>

<details><summary><h3>반응변수의 범주 간 불균형 문제</h3></summary>

- **사용 방법**

    ```
    from imblearn.over_sampling import SMOTE

    # smote 인스턴스 생성
    sm = SMOTE(random_state = 121)

    # 레코드가 부족한 범주 복제
    X_train_over, y_train_over = sm.fit_resample(X_train, y_train)

    print(f'SMOTE 적용 전 학습용 피처/레이블 데이터 세트 : {X_train.shape}, {y_train.shape}')
    print(f'SMOTE 적용 후 학습용 피처/레이블 데이터 세트 : {X_train_over.shape}, {y_train_over.shape}')
    ```

</details>

---

## 📈 회귀분석

<details><summary><h3>설명변수 간 다중공선성 문제</h3></summary>

- **다중공선성(Multicollinearity)**
    - **정의**
        - 임의의 독립변수가 종속변수에 대하여 제공하는 정보가 다른 독립변수들이 제공하는 정보에 대하여 가지는 의존성
        - 임의의 독립변수가 다중공선성이 높다면, 해당 독립변수가 제공하는 정보를 다른 독립변수들이 제공하는 정보만으로 유추할 수 있다고 판단함

    - **다중공선성의 판단**
        - **피어슨 상관계수를 통한 판단**
            - 피어슨 상관계수를 통해 설명변수 간 상관관계를 측정함
            - 둘 사이에 상관관계가 유의미하게 측정되면 다중공선성이 있다고 판단함
            - 단, 상관관계가 유의미하다고 판단하는 일정한 기준이 없음

        - **분산팽창계수(Variance Inflation Factor; VIF)를 통한 판단**
            - 다중공선성을 측정한 수치로서 그 값이 높을수록 다중공선성이 높다고 판단함
            - 통상적으로는 10을 초과하는 경우 다중공선성이 높은 편이라고 여김
        
        - **피어슨 상관계수와 분산팽창계수 비교**
            - 피어슨 상관계수는 두 변수 간 상관관계 측정에 초점을 맞춤
            - 분산팽창계수는 한 변수의 다른 변수들에 대한 의존성 측정에 초점을 맞춤
            - 따라서 분산팽창계수가 다중공선성을 판단하기에 보다 적합한 지표임
        
    - **다중공선성의 처리**
        - **피어슨 상관계수를 통한 설명변수 간 의존성 확인**
            - 일차적으로 피어슨 상관계수를 통해 다중공선성이 의심되는 변수 및 해당 변수가 의존하고 있을 것으로 의심되는 변수를 확인함
        
        - **분산팽창계수를 통한 설명변수 선별**
            - 이차적으로 분산팽창계수를 통해 다중공선성이 가장 높다고 판단된 변수를 삭제함
            - 모든 설명변수의 분산팽창계수가 10 미만이 될 때까지 반복함

- **설명변수 간 피어슨 상관계수 시각화**

    - **히트맵**

        ```
        import matplotlib.pyplot as plt
        import seaborn as sns
        %matplotlib inline

        # 설명변수 세트 X의 각 컬럼에 대하여 피어슨 상관계수 계산
        X_corr = X.astype(float).corr()

        # 히트맵 크기 설정
        plt.figure(figsize = (25, 12))

        # 팔레트 설정
        colormap = plt.cm.Reds

        # 히트맵 그리기
        sns.heatmap(
            X_corr,
            cmap = colormap,
            linewidths = 0.01, 
            linecolor = 'white', 
            vmax = 1.0, 
            vmin = -1.0,
            square = True,
            annot = True, 
            annot_kws = {"size" : 12}
            )

        plt.show()
        ```

    - **산점도**

        ```
        import matplotlib.pyplot as plt
        import seaborn as sns
        %matplotlib inline

        # 산점도 크기 설정
        plt.figure(figsize = (30, 30))

        # 산점도 그리기
        sns.pairplot(X)

        plt.show()
        ```

- **분산팽창계수를 통한 변수 선별**

    ```
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # 분산팽창계수 임계값 설정
    i = 10
    
    # 모든 설명변수의 분산팽창계수가 i 미만이 될 때까지 분산팽창계수가 가장 높은 설명변수를 제거하는 과정을 반복함
    while True :
        vif = pd.DataFrame()
        vif['feature'] = X.columns
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_max = vif['VIF'].max()
        vif_max_col = vif[vif['VIF'] == vif_max].loc[:, 'feature']
        
        if vif_max >= i : X = X.drop(vif_max_col, axis = 1)
        else : break

    # 최종 설명변수들의 분산팽창계수 확인
    print(vif)

    # 설명변수 확인
    print(X)
    ```

</details>