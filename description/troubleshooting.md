## 🫵 과적합

<details><summary><h3>overfitting & underfitting</h3></summary>

# 과대적합(Overfitting)과 과소적합(Underfitting) : test_size와 관련
![](https://gratus907.github.io/images/81b7294441f2b9c96cce938661b95a1d20d22366e5c0f72e48d2c69c9c7ad7b4.png)

- 과대적합
    - 모델이 학습 데이터에만 과도하게 최적화되어, 실제 예측을 다른 데이터로 수행할 경우에는 예측 성능이 과도하게 떨어지는 것을 말한다.

- 과소적합
    - 모델이 너무 단순하여 학습 데이터를 충분히 학습하지 못함
    - 데이터가 불충분하여 학습이 부족함

- 교차검증을 통해 과대적합, 과소적합을 방지할 수 있다!

## Model Selection 모듈이 필요한 이유
- model_selection - 학습 데이터와 테스트 데이터 세트를 분리 또는 교차 검증분할 및 평가, 하이퍼 파라미터 튜닝을 위한 다양한 함수 제공

</details>

---

## 교차검증

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

</details>