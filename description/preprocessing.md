<details><summary><h2>지도학습 순서</h2></summary>

0. **주어진 데이터 세트에 적합한 분석 방법 선택**
    - 반응변수가 범주형 변수인 경우 : 분류분석
    - 반응변수가 수치형 변수인 경우 : 회귀분석

1. **주어진 데이터 세트를 분석하기 적합한 알고리즘 선정**
    - 분류분석 알고리즘의 예시
        - `Decision Tree`, `k-NN`, `Logistic Regression`, `Ensemble` 등
    
    - 회귀분석 알고리즘의 예시
        - 선형회귀 : `Linear Regression`, `SGD Regression` 등
        - 비선형회귀 : `Decision Tree`, `Ensemble` 등

2. **해당 알고리즘의 인스턴스 생성**

3. **데이터 세트를 용도에 따라 분리**

4. **데이터 세트별 설명변수 전처리**
    - 결측치 처리
    - 수치형 설명변수 : 이상치 제거, 표준화, 정규화 등
    - 범주형 설명변수 : 인코딩 등

5. **훈련용 데이터 세트를 활용하여 인스턴스 훈련**

6. **검증용 데이터 세트를 활용하여 인스턴스 검증**
    - 검증 절차를 필수로 거쳐야 하는 것은 아님

7. **평가용 데이터 세트를 활용하여 인스턴스 성능 평가**

</details>

---

## ✂︎ 데이터 세트 나누기

<details><summary><h3>데이터 세트 나누기</h3></summary>

![](https://miro.medium.com/max/1400/0*DKB-pJy7-G6gEkM-)

- **목적 : 인스턴스 훈련에 사용할 데이터 세트와 성능 평가에 사용할 데이터 세트르 구분하기 위함**
    
- **용도**
    - **훈련용(train)** : 인스턴스 훈련(혹은 학습) 용도
    - **검증용(validation)** : 과적합을 충분히 방지하여 훈련을 중단해도 무방한지 판단 용도
    - **평가용(test)** : 인스턴스 성능 평가 용도로서 훈련 시 사용되지 않은 레코드

- **사용 방법**

```
from sklearn.model_selection import train_test_split
```

test_size: 테스트 셋 구성의 비율을 나타냅니다. train_size의 옵션과 반대 관계에 있는 옵션 값이며, 주로 test_size를 지정해 줍니다. 0.2는 전체 데이터 셋의 20%를 test (validation) 셋으로 지정하겠다는 의미입니다. default 값은 0.25 입니다.
shuffle: default=True 입니다. split을 해주기 이전에 섞을건지 여부입니다. 보통은 default 값으로 놔둡니다.
stratify: default=None 입니다. classification을 다룰 때 매우 중요한 옵션값입니다. stratify 값을 target으로 지정해주면 각각의 class 비율(ratio)을 train / validation에 유지해 줍니다. (한 쪽에 쏠려서 분배되는 것을 방지합니다) 만약 이 옵션을 지정해 주지 않고 classification 문제를 다룬다면, 성능의 차이가 많이 날 수 있습니다.
random_state: 세트를 섞을 때 해당 int 값을 보고 섞으며, 하이퍼 파라미터를 튜닝시 이 값을 고정해두고 튜닝해야 매번 데이터셋이 변경되는 것을 방지할 수 있습니다.



train_test_split()

- iris_data: 피처 데이터 세트
    - iris_label: 레이블 데이터 세트
    - test_size=0.3: 전체 데이터 세트 중 테스트 데이터 세트 비율 = 30%
    - random_state: 호출 시마다 같은 학습/테스트용 데이터 세트를 생성하기 위해 주어지는 난수 발생 값 (여기서는 값 고정을 위해 임의 숫자를 넣음)
    - train_test_split(): 호출 시 무작위로 데이터를 분리 → random_state를 지정하지 않으면 수행할 때마다 다른 학습/테스트용 데이터가 생성됨

- train_test_split() 구분

| X_train |	X_test | y_train | y_test |
|---------|--------|---------|--------|
|학습용 피처 데이터 세트|테스트용 피처 데이터 세트|학습용 레이블 데이터 세트|테스트용 레이블 데이터 세트|

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