## 📊 분류분석

<details><summary><h3>범주 간 불균형</h3></summary>

</details>

<details><summary><h3>불필요한 설명변수 선별</h3></summary>

</details>

---

## 📈 회귀분석

<details><summary><h3>설명변수 간 다중공선성</h3></summary>

- **다중공선성(Multicollinearity)이란 무엇인가**
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
    
    - 피어슨 상관계수로는 다중공선성이 의심되는 변수 및 해당 변수가 의존하고 있을 것으로 의심되는 변수를 확인함 
                - 따라서 피어슨 상관계수로 다중공선성이 의심되는 변수들을 확인한 후 다른 계수를 추가로 활용할 것을 권장함

</details>

---

## 🫵 과적합

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