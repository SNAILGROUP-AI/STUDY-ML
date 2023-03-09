# 자전거 대여 수요 예측하기

- 실습 기간 : 2022. 10. 06.

- 제출일 : 2022. 10. 07.

- 발표일 : 2022. 10. 07.

---

## 💁‍♂️ 실습 소개

<details><summary><h3>실습 목적</h3></summary>

- 지도학습 중 회귀분석 알고리즘 복습
	- 회귀분석(Regression Analysis)
		- 관찰된 연속형 변수들에 대하여 두 변수 사이의 상관관계를 규명함
		- 회귀식 : 종속변수와 독립변수(설명변수)의 상관관계를 규명한 방정식
		- 회귀선 : 회귀식을 좌표상에 표현한 선

	- 학습할 알고리즘
		
			- 선형 회귀(Linear Regressor)
			- 확률적 경사 하강 회귀(Stochastic Gradient Descent Regression; SGD Regressor)
			- 랜덤 포레스트 회귀(Random Forest Regressor)
			- 그라디언트 부스팅 회귀(Gradient Boosting Machine Regressor; GBM Regressor)
</details>

<details><summary><h3>데이터 셋 소개</h3></summary>

- 2011, 2012년 자전거 공유 시스템 이용자 수와 해당일 기후, 요일 등에 관한 정보를 담은 데이터 셋
	- 본 데이터 셋은 2015년 캐글에서 개최한 공모전에서 제공되었던 데이터 셋임
	- `train.csv` 데이터 셋을 통해 자전거 대여량 예측 모델 설계
	- `test.csv` 데이터 셋을 통해 매달 20~30일 자전거 대여량 예측
	- `sampleSubmission.csv` 데이터 셋에 예측 결과 기입

- `train.csv` : 매달 1~19일까지의 정보

	<img width="1003" alt="스크린샷 2022-10-30 오후 7 22 22" src="https://user-images.githubusercontent.com/116495744/198873751-a27f2575-8753-45aa-a3ba-aedd51dfd243.png">
    
- `test.csv` : 자전거 대여량을 제외한 매달 20~30일까지의 정보

	<img width="804" alt="image" src="https://user-images.githubusercontent.com/116495744/198872846-c11717be-754f-4c1b-9bea-14a5e9daeb28.png">
      
- `sampleSubmission.csv` : 매달 20~30일 자전거 대여량 예측값을 기입할 데이터프레임
</details>

---

## 🔎 알고리즘 소개

<details><summary><h3>선형 회귀(Linear Regression)</h3></summary>

		from sklearn.linear_model import LinearRegression
		li_reg = LinearRegression()

- 최소제곱법을 통해 회귀식을 도출하는 알고리즘

	- 최소제곱법(Ordinary Least Squares; OLS)
		- 편차 제곱의 합을 최소화하는 회귀식을 도출하는 방법
		- 편차 : 실제값과 예측값의 차이
	
	- 왜 편차를 제곱하는가?
		- 중요한 것은 편차의 방향성(음/양)이 아니라 크기임
		- 따라서 편차를 제곱함으로써 그 방향성의 영향력을 제거함

- 다음의 속성을 통해 훈련된 모델의 정보를 확인할 수 있음

			- n_features_in_ : 독립변수의 수
			- feature_nmaes_in_ : 독립변수명
			- coef_ : 각 독립변수의 가중치
			- intercept_ : 편향성
</details>

<details><summary><h3>확률적 경사 하강 회귀(Stochastic Gradient Descent Regression; SGD Regression)</h3></summary>

		from sklearn.linear_model import SGDRegressor
		sgd_reg = SGDRegressor(learning_rate = 0.1)

- 경사하강법을 통해 회귀식을 도출하는 알고리즘

	- 확률적 경사하강법(Stochastic Gradient Descent; SGD)
		- 최적화된 손실함수에 근거한 회귀식을 도출하는 방법
		- 손실함수(Loss Function) : 편차를 종속변수, 가중치를 독립변수로 가지는 함수
		- 최적화(Optimizing) : 손실함수의 결과값을 최소화하는 가중치를 찾는 일
	
	- 왜 경사를 하강하는 방법이라고 부르는가?
		- 손실함수의 결과값을 최소화하는 일계조건은 손실함수의 도함수 값이 0이 되는 것임
		- 도함수의 결과값은 좌표상으로는 원함수의 경사(Gradient)를 나타냄
		- 따라서 확률적 경사하강법은 원함수의 경사가 수평이 되는 지점을 찾는 일이라고 볼 수 있음

	- 학습률(Learning Rate)
		- 임의로 선택된 한 점에서 시작하여 한 STEP씩 움직이면서 경사를 확인함
		- 여기서 STEP의 보폭을 학습률이라고 함
		- 학습률은 정확도에 비례하고, 처리 속도에 반비례함

- 매개변수

			- learning_rate(default = 0.1) : 학습률

- 다음의 속성을 통해 훈련된 모델의 정보를 확인할 수 있음

			- n_features_in_ : 독립변수의 수
			- feature_nmaes_in_ : 독립변수명
			- coef_ : 각 독립변수의 가중치
			- intercept_ : 편향성
</details>

<details><summary><h3>랜덤 포레스트 회귀(Random Forest Regression)</h3></summary>

		from sklearn.ensemble import RandomForestRegressor
		rf_reg = RandomForestRegressor(n_estimators = 100)

- 앙상블 기법 중 의사결정나무 회귀(Decision Tree Regression) 알고리즘에 배깅 방식이 적용된 알고리즘
	- 앙상블(Ensemble) 기법 : 여러 모델을 조합한 하나의 모델을 만드는 기법
	- 배깅(Bagging) : 여러 알고리즘을 병렬로 학습시키고, 결과의 대표값(평균)을 예측값으로 반환하는 방식

- 매개변수

			- n_estimators(default = 100) : 동원할 모델의 수

- 다음의 속성을 통해 훈련된 모델의 정보를 확인할 수 있음

			- estimators_ : 동원된 모델 목록
			- n_features_ : 독립변수의 수
			- feature_names_in_ : 독립변수명
			- feature_importances_ : 각 독립변수의 가중치
</details>

<details><summary><h3>그라디언트 부스팅 회귀(Gradient Boosting Machine Regression; GBM Regression)</h3></summary>

		from sklearn.ensemble import GradientBoostingRegressor
		gbm_reg = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.1)

- 앙상블 기법 중 의사결정나무 회귀(Decision Tree Regression) 알고리즘에 부스팅 방식이 적용된 알고리즘	
	- 부스팅(Boosting)
		- 여러 모델을 직렬로(혹은 순차로) 학습시키는 방식
		- 이전 순번 모델에서 편차가 큰 부분에 다음 순번 모델이 가중치를 두어 학습하도록 함
	
	- 학습률(Learning Rate)
		- 편차가 큰 부분에 대하여 부여할 가중치
		- 그라디언트 부스팅 회귀 알고리즘은 편차를 경사하강법으로 계산함

- 매개변수

			- n_estimators(default = 100) : 동원할 모델의 수
			- learning_rate(default = 0.1) : 학습률

- 다음의 속성을 통해 훈련된 모델의 정보를 확인할 수 있음
			
			- estimators_ : 동원된 모델 목록
			- n_features_ : 독립변수의 수
			- feature_name_in_ : 독립변수명
			- feature_importances_ : 각 독립변수의 가중치
</details>

<details><summary><h3>성능평가지표</h3></summary>

- 결정계수(coefficient of determination; r2-score)
	- 실제 값의 분산 대비 예측 값의 분산 비율
	- 0~1 사이의 값을 가지며, 값이 클수록 회귀식의 적합도가 높다고 판단함

- 평균제곱편차(Mean Squared Error; MSE)
	- 오차(실제 값과 예측 값의 차이)를 제곱한 값의 평균
	- 값이 작을수록 회귀식의 적합도가 높다고 판단함
	- 오차를 제곱하므로 값을 과장할 수 있음

- 평균제곱근편차(Root Mean Squared Error; RMSE)
	- 평균제곱편차의 제곱근
	- 평균제곱편차에 제곱근하는 절차를 더하여 오차의 크기가 과장된 정도를 줄임

- 평균절대편차(Mean Absolute Error; MAE)
	- 오차(실제 값과 예측 값의 차이)의 절대값의 평균
	- 평균제곱편차에서 오차를 제곱하는 이유는 값의 방향성(음/양)이 아니라, 크기가 중요하기 때문임
	- 따라서 오차를 제곱한 값 대신 오차의 절대값을 활용하여 오차의 크기가 과장될 여지를 없앰
</details>

---

## 📄 META-DATA

<details><summary><h3>Feature Columns</h3></summary>

- 명목변수
  - `Datetime` : 이용 시각      

        - YYYY-MM-DD 00:00:00

  - `Season` : 계절
     
        - 봄(1)
        - 여름(2)
        - 가을(3)
        - 겨울(4)

  - `Holiday` : 공휴일 여부
    
        - 공휴일(1) : '주말을 제외한' 주중 쉬는 날
        - 그 외(0)

  - `Workingday` : 근무일 여부
    
        - 근무일(1)
        - 그 외(0)

  - `Weather` : 날씨

        - 아주 깨끗한 날씨(1)
        - 약간의 안개와 구름(2)
        - 약간의 비와 눈(3)
        - 아주 많은 비와 우박(4)

- 실질변수
  - `Temp` : 온도(섭씨)
  
  - `Atemp` : 체감 온도(섭씨)
  
  - `Humidity` : 습도
  
  - `Windspeed` : 풍속
</details>

<details><summary><h3>Target Column</h3></summary>

- `Count` : 총 자전거 대여량
    
      - `Count` = `Casual` + `Registered`
      - `Casual` : 비회원 자전거 대여량
      - `Registered` : 회원 자전거 대여량
</details>

---

## ✨ PRE-PROCESSING

<details><summary><h3>데이터 셋 탐색하기</h3></summary>

- `datetime` : 자료형이 object임

- `season` : 통상적인 기준과는 다르게 분류함

- `datetime`의 날짜 : `train.csv`와 `test.csv` 간 고르게 분포되어 있지 않음

- `casual` : `test.csv` 및 `smapleSubmission.csv`에는 존재하지 않음

- `registered` : `test.csv` 및 `smapleSubmission.csv`에는 존재하지 않음
</details>

<details><summary><h3>칼럼 재정의하기</h3></summary>

- `datetime` : 년/월/일/시/요일로 파싱함

        - year : 2011, 2012
        - month : 1, 2, 3, ..., 12
        - day : 1, 2, 3, ..., 19
        - hour : 0, 1, 2, ..., 23
        - weekday : 월(0), 화(1), 수(2), 목(3), 금(4), 토(5), 일(6)

- `season` : 재분류

      - 봄(0) : 3, 4, 5월
      - 여름(1) : 6, 7, 8월
      - 가을(2) : 9, 10, 11월
      - 겨울(3) : 12, 1, 2월

- `day`, `casual`, `registered` : 삭제
</details>

<details><summary><h3>독립변수 간 다중공선성 줄이기</h3></summary>

- 히트맵을 통한 상관계수 확인

	![상관계수](https://user-images.githubusercontent.com/116495744/199893643-db3680a5-e38e-4fc3-ada7-9f9ee4ccb269.png)

	- `atemp`와 `temp`의 상관계수가 약 0.98로 매우 높은 의존성을 보임
	- 두 독립변수 간 상관관계가 높다면 각 독립변수가 종송변수에 대하여 설명하는 정보에 유의미한 차이가 있다고 볼 수 없음
	- 따라서 두 변수 중 다중공선성이 더 높은 변수를 삭제하기로 결정함

- 다중공선성
	- 다중공선성(Multicollinearity)
		- 임의의 독립변수가 종속변수에 대하여 제공하는 정보가 다른 독립변수들이 제공하는 정보에 대하여 가지는 의존성
		- 임의의 독립변수가 다중공선성이 높다면, 해당 독립변수가 제공하는 정보를 다른 독립변수들이 제공하는 정보만으로 유추할 수 있다고 판단함
	
	- 분산팽창계수(Variance Inflation Factor; VIF)
		- 다중공선성을 측정한 수치
		- 분산팽창계수가 높을수록 다중공선성이 높다고 판단함
		- 통상적으로는 10을 초과하는 경우 다중공선성이 높은 편이라고 여김

- 분산팽창계수 확인 및 무의미한 독립변수 삭제
	
	<img width="244" alt="스크린샷 2022-11-04 오전 2 26 16" src="https://user-images.githubusercontent.com/116495744/199792562-1280e09d-e44b-4c5c-8ba3-2d21ed5a16a8.png">
	
	- `atemp`의 분산팽창계수는 317.0, `temp`의 분산팽창계수는 277.0으로, `atemp`의 다중공선성이 더 높음
	- 그밖에 `year`의 분산팽창계수는 약 79.0, `humidity`의 분산팽창계수는 약 19.0으로, 다중공선성이 높다고 볼 수 있음
	- 모든 독립변수들의 분산팽창계수가 10 이하가 될 때까지, 다중공선성이 가장 높은 독립변수부터 하나씩 삭제함
	- 최종적으로 총 8개 독립변수가 잔존함

				실질변수 : temp, windspeed
				명목변수 : weather, month, hour, weekday, holiday, workingday
</details>

<details><summary><h3>실질변수 전처리</h3></summary>

- boxplot을 통한 실질변수 분포 탐색

	![이상치](https://user-images.githubusercontent.com/116495744/199793429-6159de93-8a4e-4b2d-8ea5-ea6e3563c68c.png)	

	- `windspeed`에 이상치가 존재함을 확인할 수 있었음
	- Turkey Fence 기법을 통해 이상치를 판별 및 처리하기로 결정함

- Turkey Fence 기법
	- 사분위 범위(InterQuartile Range; IQR)을  활용하여 이상치를 판별하는 기법
		
			- 사분위 범위(IQR) : 제3사분위수(Q3) - 제1사분위수(Q1)
	
	- 상한값을 초과하거나 하한값에 미달한 값을 이상치로 판별함
		
			- 하한값(lower_value) : Q1 - IQR * 1.5
			- 상한값(upper_value) : Q3 + IQR * 1.5

- `windspeed` 이상치 판별 및 처리
	- 총 10886건 중 277건이 `windspeed`에 대하여 이상치를 가지고 있음
	- 해당 자료의 `windspeed` 값을 하한값 혹은 상한값으로 대체함

- 실질변수의 분포를 회귀분석에 적합한 형태로 변환하기 위하여 표준화함
		
		- 표준화(standard scaling) : 값의 분산을 기준으로 평균 0, 분산 1인 분포로 변환하는 작업
	
- 실질변수 간 범위의 상대적 크기가 미치는 영향력을 최소화하기 위하여 정규화함
	
		- 정규화(minimax scaling) : 값의 크기를 기준으로 최솟값을 0, 최댓값을 1인 분포로 축소하는 작업
</details>

<details><summary><h3>명목변수 전처리</h3></summary>

- 이미 모든 명목변수가 Label Encoding되어 있음

- One-Hot Encoding을 통해 다음과 같이 이진분류 칼럼으로 파싱함

		- season : season_0, season_1, season_2, season_3
		- month : month_1, month_2, month_3, ..., month_12
		- hour : hour_0, hour_1, hour_2, ..., hour_23
		- weekday : weekday_0, weekday_1, weekday_2, ..., weekday_6
		- holiday : holiday_0, holiday_1
		- workingday : workingday_0, workingday_1
		- weather : weather_1, weather_2, weather_3, weather_4
</details>

<details><summary><h3>Pipeline 만들기</h3></summary>

</details>

---

## 🚀 MODEL DESIGN

<details><summary><h3>모델 설계</h3></summary>

</details>

<details><summary><h3>모델 선정</h3></summary>

</details>

<details><summary><h3>test.csv 예측</h3></summary>

</details>

<details><summary><h3>예측 결과 분석</h3></summary>

![다운로드](https://user-images.githubusercontent.com/116495744/200706287-a05d51e0-90e9-4779-bf9b-0e4783a3ecb8.png)
![7](https://user-images.githubusercontent.com/116495744/200705859-68d6e437-c27b-408e-8066-ca986611d0f4.png)
![6](https://user-images.githubusercontent.com/116495744/200705864-930cc01f-954e-4059-846e-f3dd20c7a59d.png)
![5](https://user-images.githubusercontent.com/116495744/200705866-bb5d814c-0690-4e09-8c38-7e0f1e4f5f1d.png)
![4](https://user-images.githubusercontent.com/116495744/200705868-338b132e-ce33-4641-9001-3b2be051d289.png)
![3](https://user-images.githubusercontent.com/116495744/200705873-a984292f-f90e-482d-9646-032c36a4ee3f.png)
![2](https://user-images.githubusercontent.com/116495744/200705874-619c4876-c4a6-492c-9693-d89cd808887d.png)
</details>

---

## :lips: COMMENT

<details><summary><h3>Teacher`s</h3></summary>

> **지금까지 배운 내용들을 총망라했다고 생각합니다. 짧은 시간 안에 다양한 방법으로 데이터를 분석하고, 적합한 모델을 찾기 위해 여러 알고리즘을 적용한 점에서 팀원들 간 협업도와 각각의 몰입도를 느낄 수 있었습니다. 또한 목차를 논리정연하게 정리한 점, 코드를 이해 가능하도록 기술한 점, scoresDF라는 데이터프레임을 생성하여 각 모델의 성능 지표를 한번에 비교할 수 있도록 기획한 점이 인상 깊었습니다.**
</details>

<details><summary><h3>Students`</h3></summary>

> 데이터 처리 과정부터 머신러닝 모델 설계 및 예측 과정까지 일목요연하게 기술한 점, 짧은 시간 안에 여러 방법을 시도한 점,  모델 선정 과정에서 각각의 성능 지표를 한번에 확인 가능하도록 기획한 점이 인상 갚었습니다.

<br>

> 데이터 전처리를 단순히 수행한 것만이 아니라 왜 그러한 방식으로 수행했는지 설명하고, 또한 모델 선정 과정에서 왜 해당 모델을 선정했는지 납득할 수 있도록 각 모델의 성능지표를 한번에 비교할 수 있는 데이터프레임을 보여준 점이 좋았습니다.

<br>

> 데이터 전처리 과정을 설명할 때 해당 데이터에 대한 이해도를 엿볼 수 있었습니다. 또한 짧은 시간 안에 다양한 모델을 구상하고 서로 비교할 수 있었음에 놀라웠습니다.

<br>

> 짧은 시간 안에 다양한 알고리즘을 적용하고 각각의 성능지표를 한번에 비교할 수 있도록 정리한 점이 인상 깊습니다.

<br>

> 다양한 알고리즘을 적용하고, 각각의 성능을 데이터프레임으로 정리한 점을 포함하여 전반적으로 깔끔하고 이해하기 수월했습니다.

<br>

> 지금까지 학습한 내용을 모두 반영한 프로젝트라고 생각합니다. 데이터를 전처리하거나 알고리즘에 적용하는 과정에서 많은 고민의 흔적을 느낄 수 있었습니다. 또한 여러 모델을 적용하고 성능을 비교한 점이 인상 깊었습니다.

<br>

> 메타데이터에 대한 자세한 설명, 계절을 실제에 부합하도록 전처리한 점, 다양한 모델을 구상한 점이 인상 깊었습니다.

<br>

> 짧은 시간 안에 다양한 시도를 한 점이 놀랍습니다. 뿐만 아니라 전처리 과정을 이해하기 쉽게 설명한 점이나 성능지표에 관한 데이터프레임을 기획한 점 등에서 정리정돈이 잘 되었음을 느낄 수 있었습니다.
</details>

---

## 👭 WORKMATE

👨 [**IT`S ME**](https://github.com/jayarnim)

    - Exploratory Data Analysis
    - README

👩 [**김효정**](https://github.com/410am)

    - LinearRegression
    - SGDRegresion
    - RandomForestRegression
    - GradientBoostingRegression

👨 [**인찬휘**](https://github.com/wassaa-1)

    - Exploratory Data Analysis
    - Predict
    - Presentation

<br>

---

## 🛠 SKILL USED

### 👉 LANGUAGE

<img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white"/>

### 👉 IDE

<img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=Google Colab&logoColor=white"/>

### 👉 LIBRARY

<img src="https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white"/> &nbsp;
<img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/> &nbsp;
<img src="https://img.shields.io/badge/scikitlearn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>

<br>

---

## 💡 자료 출처

- [**bike-sharing-demand (kaggle)**](https://www.kaggle.com/c/bike-sharing-demand/overview)
