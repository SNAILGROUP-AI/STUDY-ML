## ✂︎ 데이터 세트 나누기

<details><summary><h3>데이터 세트 나누기</h3></summary>

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

## 🔢 Numerical Feature Engineering

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
    
    col = "이상치를 처리할 컬럼명"
    before_scaled = X[[col]]

    # Turkey Fence 기법에 기반한 이상치 탐지 및 처리기 RobustScaler 인스턴스 생성
    scaler = RobustScaler()

    # 이상치 탐지
    scaler.fit(before_scaled)

    # 이상치 처리
    after_scaled = scaler.transform(before_scaled)

    # 이상치 처리 전후 비교
    before_scaled = before_scaled.rename(columns = {col : "before"})
    after_scaled = after_scaled.rename(columns = {col : "after"})
    scale_df = pd.concat([before_scaled, after_scaled], axis = 1)

    print(scale_df)
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
    
    col = "표준화할 컬럼명"
    before_scaled = X[[col]]

    # 표준화 처리기 StandardScaler 인스턴스 생성
    scaler = StandardScaler()

    # 평균 및 분산 탐색
    scaler.fit(before_scaled)

    # 표준화
    after_scaled = scaler.transform(before_scaled)

    # 표준화 전후 비교
    before_scaled = before_scaled.rename(columns = {col : "before"})
    after_scaled = after_scaled.rename(columns = {col : "after"})
    scale_df = pd.concat([before_scaled, after_scaled], axis = 1)

    print(scale_df)
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
    
    col = "정규화할 컬럼명"
    before_scaled = X[[col]]

    # 정규화 처리기 MinMaxScaler 인스턴스 생성
    scaler = MinMaxScaler()

    # 최대최소 변환을 위한 분포 탐색
    scaler.fit(before_scaled)

    # 정규화
    after_scaled = scaler.transform(before_scaled)

    # 정규화 전후 비교
    before_scaled = before_scaled.rename(columns = {col : "before"})
    after_scaled = after_scaled.rename(columns = {col : "after"})
    scale_df = pd.concat([before_scaled, after_scaled], axis = 1)

    print(scale_df)
    ```

</details>

<details><summary><h3>수치형 설명변수의 전처리 순서</h3></summary>

![스케일링 비교](https://miro.medium.com/max/1400/1*0Ox-p57oxfmaVSaJyJWyPg.png)

- **`RobustScaler` 👉  `StandardScaler` 👉 `MinMaxScaler` 순을 권장함**

    - 이상치가 존재할 경우 정규화에 따른 성능 개선 효과가 미미함
    - 정규화 이후 표준화를 하는 경우 설명변수별 범위가 재조정될 가능성이 있음

</details>

---

## 🔤 Categorical Feature Engineering

<details><summary><h3>레이블 인코딩</h3></summary>

- **사용 방법**

    ```
    from sklearn.preprocessing import LabelEncoder

    col = "인코딩할 컬럼명"
    before_encoded = X[[col]]

    # 레이블 인코더 LabelEncoder 인스턴스 생성
    label = LabelEncoder()

    # 범주 탐색
    label.fit(before_encoded)

    # 레이블 인코딩
    after_label = label.transform(before_encoded)
    
    # 레이블 인코딩 전후 비교
    before_encoded = before_encoded.rename(columns = {col : "before"})
    after_label = after_label.rename(columns = {col : "label"})
    encode_df = pd.concat([before_encoded, after_label], axis = 1)

    print(encode_df)
    ```

- **다음을 통해 인코더의 정보를 확인할 수 있음**
    - `classes_` : 숫자별 매칭되어 있는 범주명
    - `inverse_transform(xs)` : 리스트 $xs$에 대하여 그 원소들을 순차로 역인코딩하여 리스트에 저장한 후 해당 리스트를 반환함

</details>

<details><summary><h3>원 핫 인코딩</h3></summary>

- **사용 방법**

    ```
    from sklearn.preprocessing import OneHotEncoder

    # 원 핫 인코더 OneHotEncoder 인스턴스 생성
    oht = OneHotEncoder()

    # 레이블 인코딩한 3차원 행렬 after_label을 2차원 벡터로 변환
    before_oht = after_label.reshape(-1, 1)

    # 범주 탐색
    oht.fit(before_oht)

    # 원 핫 인코딩
    after_oht = oht.transform(before_oht)

    # 결과를 희소행렬 형태에서 밀집행렬 형태로 변환
    after_oht = after_oht.toarray()


 
    before_scaled = before_scaled.rename(columns = {col : "before"})
    after_scaled = after_scaled.rename(columns = {col : "after"})
    encode_df = pd.concat([before_scaled, after_scaled], axis = 1)

    print(scale_df)

    print(scale_df)
    for col in cat_col :
        xs = df[col]
        
        label = LabelEncoder()
        xs = label.fit_transform(xs)
        
        xs = xs.reshape(-1, 1)
        
        oht = OneHotEncoder()
        xs = oht.fit_transform(xs)
        
        xs = xs.toarray()

        label_list = list(label.classes_)
        label_list = [col + "_" + label_list[i] for i in range(len(label_list))]
        
        encoded_col = pd.DataFrame(xs, columns = label_list)
        encoded_list.append(encoded_col)

    encoded_df = pd.concat(encoded_list, axis = 1)
    df = pd.concat([df, encoded_df], axis = 1)
    df = df.drop(columns = cat_col)
    ```


</details>

---

## ☑️ Feature Selecting

<details><summary><h3>분류분석 - 승산비 기준</h3></summary>

- **승산비의 이해**
    - **승산(odds)**
        - 이항범주형 반응변수에 대하여 반응하지 않을 가능성($1-p$) 대비 반응할 가능성($p$)
        - 반응변수가 반응할 가능성이 반응하지 않을 가능성보다 몇 배 높은가
        - 반응변수가 반응할 가능성을 $p$ 라고 했을 때, 승산 $odds$ 는 다음과 같음
        
        ### $$odds=\frac{p}{1-p}$$

    - **승산비(Oods Ratio; OR)**
        - 이항범주형 반응변수 y와 이항범주형 설명변수 x에 대하여 x의 변동에 따른 y의 반응
        
        - 설명변수가 참일 때 반응변수가 반응할 가능성이 거짓일 때보다 몇 배 높은가
        
        - 이항범주형 반응변수 y와 이항범주형 설명변수 x에 대하여 다음과 같이 가정하자
            - x가 참일 때 y가 반응할 확률 : $a$
            - x가 참일 때 y가 반응하지 않을 확률 : $b$
            - x가 거짓일 때 y가 반응할 확률 : $c$
            - x가 거짓일 때 y가 반응하지 않을 확률 : $d$
            - $a+b+c+d=1$
        
        - x에 대한 y의 승산비 $OR$ 은 다음과 같음
        
        ### $$OR=\frac{a/b}{c/d}$$
    
    - **승산비의 해석**
        - $OR \approx 1$ : 해당 설명변수와 반응변수 간 상관관계가 유의미하지 않다고 판단함
        - $OR < 1$ : 해당 설명변수와 반응변수 간 음의 상관관계가 있다고 판단함
        - $OR > 1$ : 해당 설명변수와 반응변수 간 양의 상관관계가 있다고 판단함

- **로지스틱 회귀식의 가중치의 이해**
    - 단순회귀분석 하의 로지스틱 회귀식은 다음과 같음

    ### $$ln(\frac{p}{1-p})=w_0+wX$$
    
    - 이항범주형 반응변수 y와 이항범주형 설명변수 X에 대하여 다음과 같이 가정하자
        - x가 참일 때 y가 반응할 확률 : $a$
        - x가 참일 때 y가 반응하지 않을 확률 : $b$
        - x가 거짓일 때 y가 반응할 확률 : $c$
        - x가 거짓일 때 y가 반응하지 않을 확률 : $d$
        - $a+b+c+d=1$

    - X가 참(1)일 때의 회귀식은 다음과 같음
    
    ### $$ln(\frac{a}{b})=w_0+w$$

    - X가 거짓(0)일 때의 회귀식은 다음과 같음

    ### $$ln(\frac{c}{d})=w_0$$

    - 두 회귀식을 빼면 다음과 같음

    ### $$ln(\frac{a/b}{c/d})=w$$

    - 따라서 X에 대한 y의 승산비와 X의 가중치 w 간에는 다음의 관계가 성립함

    ### $$ln(\frac{a/b}{c/d})=w$$

- **결론**
    - 승산비의 신뢰구간에 1이 존재하는 경우 해당 설명변수의 변동이 반응변수에 미치는 영향력이 유의미하지 않다고 판단함
    - 즉, 로지스틱 회귀식에서 임의의 설명변수 x의 가중치 $w$ 를 지수로 가지는 지수함수 $f(w)=e^w$ 의 값에 대하여
    - 그 신뢰구간에 1이 존재하는 경우 해당 설명변수가 반응변수에 미치는 영향력이 유의미하지 않다고 판단함

- **사용 방법**

    ```
    from sklearn.linear_model import LogisticRegression
    import scipy.stats as st

    # 로지스틱 회귀 알고리즘 인스턴스 생성
    lg_clf = LogisticRegression()

    # 로지스틱 회귀분석 수행
    lg_clf.fit(X, y)

    # 설명변수, 가중치, 승산비 정보를 담은 데이터프레임 or_df 생성
    features = list(lg_clf.feature_names_in_)
    weights = list(lg_clf.coef_)
    odds_ratio = [np.exp(weights[i]) for i in range(weights)]

    or_dict = {
        'feature' : features,
        'weight' : weights,
        'or' : odds_ratio
    }

    or_df = pd.DataFrame(or_dict, index = 'feature')

    # 신뢰수준 설정
    i = 0.95

    # 95% 신뢰수준 하에서 설명변수별 승산비의 신뢰구간 확인
    # or_df에 승산비의 최소치와 최대치 정보를 담은 칼럼 추가
    or_min_list = []
    or_max_list = []

    for feature in features :
        ci = st.norm.interval(
                alpha = i, 
                loc = or_df.loc[feature, 'or'], 
                scale = st.sem(X[feature])
                )
        
        or_min_list.append(ci[0])
        or_max_list.append(ci[1])
    
    or_df['or_min'] = or_min_list
    or_df['or_max'] = or_max_list

    # 신뢰구간에 1이 포함되어 있는지 확인
    # or_df에 신뢰구간에 1 포함 여부 정보를 담은 칼럼 추가
    drop_list = []

    for feature in features :
        if (or_df.loc[feature, 'or_min'] <= 1) and (or_df.loc[feature, 'or_max'] >= 1) : drop_list.append(True)
        else : drop_list.append(False)
    
    or_df['drop'] = drop_list

    print(or_df)
    ```

</details>

<details><summary><h3>선형회귀분석 - 다중공선성 기준</h3></summary>

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

---

## 📊 Target Engineering

<details><summary><h3>분류분석 시 반응변수 레이블 간 레코드 불균형 문제</h3></summary>

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