## 🤓 군집 분석(Cluster Analysis)이란 무엇인가

- **정의**
    - 주어진 데이터 세트를 레코드 간 유사성과 상이성을 계산하여 k개의 군집으로 분류하는 작업

- **지도학습의 분류 분석과 비교**
    - **공통점**
        - 범주를 주요 논점으로 삼고 있음
    
    - **차이점**
        - 군집 개수와 구조를 파악 가능 여부
        - 분류 분석은 사전에 파악된 군집의 개수와 구조에 근거하여 레코드의 범주를 찾는 작업임
        - 군집 분석은 레코드의 특성을 고려하여 군집의 개수와 구조를 설계하는 작업임

- **구분**
    - **군집의 정의에 따른 구분**
        - **중심점(Center) 기반 군집화** : 모든 레코드는 자신이 속한 군집의 중심점에 더 가까이 위치함
        - **밀도(Density) 기반 군집화** : 군집은 높은 밀도를 가진 레코드들의 공간임
        - **근접성(Contiguity) 기반 군집화**
        - **공유된 특성(Shared Property) 기반 군집화**

    - **중첩 여부에 따른 구분**
        - **배타적 군집화(Exclusive Clustering)** : 하나의 레코드를 하나의 군집에만 배타적으로 분류하는 군집화
        - **중첩적 군집화(Overlapping Clustering)** : 하나의 레코드를 여러 군집에 중첩적으로 분류하는 군집화
    
    - **계층화 여부에 따른 구분**
        - **계층적 군집화** : 군집 간 위계가 존재하는 군집화
        - **분할적 군집화** : 군집 간 위계가 존재하지 않는 군집화

---

## k-Means

<details><summary><h3>k-Means 란 무엇인가</h3></summary>

- **정의**
    - 중심점 기반 배타적, 분할적 군집화 알고리즘

- **목표**
    - 각 군집의 Means를 최소화하는 것

        - `k` : k개의 군집
        - `Means` : 중심점과 레코드 간 평균 거리
        - `centroid` : 중심점

- **과정**
    - 데이터 세트를 그래프에 묘사함
    - `k` 개의 `centroid` 을 그래프 상에 임의로 배치함
    - 레코드를 가장 가까이 위치한 `centroid` 의 군집으로 군집화함
    - `Means` 를 계산하여 `centroid` 의 위치를 군집의 중심으로 재배치함
    - 레코드를 가장 가까이 위치한 `centroid` 의 군집으로 재군집화함
    - `Means` 를 계산하여 `centroid` 의 위치를 군집의 중심으로 재배치함
    - `Means` 가 최소화될 때까지 이상의 절차를 반복함

</details>

<details><summary><h3>SK-Learn의 k-Means 알고리즘</h3></summary>

- **사용 방법**

    ```
    ```

- **주요 하이퍼파라미터**

- **다음의 속성을 통해 훈련된 모델의 정보를 확인할 수 있음**

- **최적의 k 구하기**

</details>

---

## DBSCAN(Density-Based Spatial Clustering of Applications with Noise)

<details><summary><h3>DBSCAN 이란 무엇인가</h3></summary>

- **정의**
    - 밀도 기반 배타적, 분할적 군집화 알고리즘
    - 잡음(Noise)을 활용한 밀도 기반(Density-Based) 공간(Spatial) 군집화

- **잡음(Noise)을 활용한 밀도 기반(Density-Based) 공간(Spatial) 군집화**
    
    - **K-Means 군집 분석의 한계점**
        - 이상치를 탐지할 수 없어 해당 값에 의해 `centroid` 가 결정될 수 있음
    
    - **DBSCAN 군집 분석의 보완 방안**
        - 특정 레코드가 특정 군집에 속하는 경우, 해당 군집에 속하는 다른 레코드들과 가까운 위치에 있어야 함을 전제함

- **DBSCAN 의 레코드 구분**

    ![IMG_7115](https://user-images.githubusercontent.com/116495744/224615745-cd9d88fe-c4d4-4f90-9d8c-a989a8ffff3d.PNG)

    - **핵심 요소(Core)** : 밀도의 중심이 되는 레코드
        - **최소 요소(Minimum number of neighbors)** : 핵심 요소 지정 조건으로서 해당 요소의 직경 내에 레코드가 몇 개 존재해야 하는가
        - **직경(Radius)** : 핵심 요소 기준 반경으로서 밀도 영역(Dense Area) 혹은 군집의 범위

    - **경계 요소(Border)** : 핵심 요소 반경의 경계선에 위치한 레코드
    - **잡음 요소(Noise)** : 어떠한 군집에도 속하지 않는 레코드로서 이상치

</details>

<details><summary><h3>SK-Learn의 DBSCAN 알고리즘</h3></summary>

- **사용 방법**

    ```
    ```

- **주요 하이퍼파라미터**

- **다음의 속성을 통해 훈련된 모델의 정보를 확인할 수 있음**

</details>

---

## 💯 평가 지표

<details><summary><h3>V-measure</h3></summary>

- **군집 분석의 목표**
    - 군집 간 거리는 멀고, 군집 내 레코드 간 거리는 가깝게 군집화하는 것

- **V-measure 의 정의**
    
    - 균질성과 완전성의 조화 평균
        - **균질성(Homogeneity)** : 각 군집(예측값)이 동일한 실제값으로 구성되어 있는 정도
        - **완전성(Completeness)** : 각 실제값에 대하여 동일한 군집(예측값)으로 구성되어 있는 정도

- **전제**
    - 군집이 사전에 정의되어 있는 경우 사용함
    - 군집이 사전에 정의되어 있지 않을 경우에는 후술할 실루엣 계수를 사용함

- **사용 방법**

    ```
    from sklearn.metrics import homogeneity_score
    from sklearn.metrics import completeness_score
    from sklearn.metrics import v_measure_score
    ```

</details>

<details><summary><h3>실루엣 계수(Silhouette Coefficient)</h3></summary>

- **군집 분석의 목표** : 군집 간 거리는 멀고, 군집 내 레코드 간 거리는 가깝게 군집화하는 것

- **실루엣 계수(Silhouette Coefficient)의 정의**

- **사용 방법**

</details>

---

## 📝 Practice

- [**실습 코드**]()