## ✂︎ 데이터 셋 나누기

---

## 🔎 설명변수 탐색

---

## ❓ 결측치 처리

<details><summary><h3>결측치(Missing Value)란 무엇인가</h3></summary>

- **정의 : 기입되지 않은 데이터**

- **`None` 과 `NaN` 의 구분**
    - `None`(Null) : 아무것도 존재하지 않는 데이터
    - `NaN`(Not a Number) : 정의되거나 표현되지 못하는 데이터

- **발생 원인에 따른 구분**
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

<details><summary><h3>결측치 탐색 및 처리</h3></summary>
    
- **메소드 `isnull()`를 통한 결측치 탐색**

    ```
    # 결측치가 존재할 경우 True, 존재하지 않을 경우 False를 기입한 데이터프레임 반환
    df.isnull()

    # 각 컬럼별 결측치 개수 반환
    df.isnull().sum()

    # 각 컬럼별 결측치 비율 반환
    df.isnull().mean()
    ```

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

<details><summary><h3>단위가 들쑥날쑥한 경우</h3></summary>

- **정규화(Normalization)**
    - 정의 : 값의 범위를 특정하고 모든 설명변수의 분포를 해당 범위로 확대 혹은 축소함
    - 목적 : 모든 설명변수의 크기를 통일하여 설명변수 간 상대적 크기가 주는 영향력을 최소화함
    - 통상적으로는 최대값을 1, 음수가 존재하면 최소값을 -1, 존재하지 않으면 최소값을 0으로 변환함
    
    ### $$x_{new}=\frac{x_i-min(x)}{max(x)-min(x)}$$

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

<details><summary><h3>분포가 들쑥날쑥한 경우</h3></summary>

- **표준화(Standardization)**
    - 정의 : 값의 분포를 평균이 0, 분산이 1인 표준정규분포(가우시안 정규 분포) 형태로 변환함
    - 목적 : 모든 설명변수의 형태를 통계 분석의 가정에 부합하는 형태로 변환함

    ### $$x_{new}=\frac{x_i-mean(x)}{std(x)}$$

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

<details><summary><h3>수치형 변수의 전처리 순서</h3></summary>

![스케일링 차이](https://user-images.githubusercontent.com/116495744/222521330-0df348d5-05a0-4a45-9c4c-5591550ff2d2.jpeg)

- 이상치가 존재할 경우 정규화 과정에서 분포가 비정상적으로 촘촘해질 가능성이 높음
- 따라서 `RobustScaler` 👉  `StandardScaler` 👉 `MinMaxScaler` 순으로 스케일링할 것을 권장함

</details>

---

## 🔤 범주형 설명변수의 전처리

<details><summary><h3>인코딩</h3></summary>

</details>