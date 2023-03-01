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

<details><summary><h3>Hard Voting</h3></summary>

</details>

<details><summary><h3>Soft Voting</h3></summary>

</details>

---

## 🤝 Bagging

<details><summary><h3>Random Forest</h3></summary>

</details>

---

## 🕵️ Boosting

<details><summary><h3>Gradient Boosting Machine(GBM)</h3></summary>

</details>

<details><summary><h3>eXtra Gradient Boosting(XGBoost)</h3></summary>

</details>

<details><summary><h3>Light Gradient Boosting Machine(LightGBM)</h3></summary>

</details>