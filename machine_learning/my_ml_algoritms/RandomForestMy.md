## Случайный лес (беггинг)

Идея беггинга заключается в том чтобы построить композицию моделей с меньшим относительно одной модели разбросом. Построение композиции происходит за счет генерации бутстрепом выборки $\widetilde X$ из всей имеющейся выборки $X$ (где $|\widetilde X| = d$, размеру выборки) и построения модели $b(x)$ с помощью модели $\mu$ на выборке $\widetilde X$.
$$ a(x) = \frac{1}{N}\sum_{n=1}^N b_n(x)$$ 
где $a(x)$ - композиция моделей.

Для того чтобы понять принцип работы беггинга, разложим ошибку предсказания на компоненты
$$L(\mu) = \mathbb E_{x, y}(y-\mathbb E(y|x))^2 + \mathbb E_x (\mathbb E_x \mu(X) - \mathbb E(y|x))^2 + \mathbb E_x \mathbb E_X(\mu(X) - \mathbb E_x \mu(X))^2 $$
где 
* 1-я компонента в формуле - шумовая компонента: показывает насколько плох лучший из возможных алгоритмов
* 2-я компонента - смещение (bias): показывает насколько в среднем мы в состоянии приблизить лучшую модель
* 3-я компонента - разброс (variance): отклонение модели на данной выборке относительно средней модели

Воздействовать на 1-ю компоненту мы не можем по определению, на 2-ю в случае беггинга не получится в силу того что смещение композиции $a(x)$ равно смещению любой из модели $b(x)$, т.е. не уменьшится от количества моделей и зависит от выбора базовой модели (необходимо выбирать базовую модель с низким смещением), а вот влияние на 3-ю разберем подробнее. Формула разброса выглядит следующим образом:
$$variance(a(x)) = (\frac{1}{N} \mathbb E_x \mathbb E_X (\tilde \mu (X) - \mathbb E_X \tilde \mu (X))^2)(1) + (\frac{N*(N-1)}{N^2} \mathbb E_x \mathbb E_X (\tilde \mu_1(X) - \mathbb E_x \tilde \mu_1 )(\tilde \mu_2(X) - \mathbb E_x \tilde \mu_2 ))(2)$$
где 
* $\tilde \mu (X) - \mathbb E_X \tilde \mu (X)$ - разброс одной модели
* $\mathbb E_x$ - перебор по всем объектам множества X
* $\mathbb E_X$ - перебор по всем подмножествам
* $(\tilde \mu_1(X) - \mathbb E_x \tilde \mu_1 )$ - ковариация моделей построенных бутсрепом

Из формулы разброса делаем вывод что для минизации разброса необходимо:
1) Максимально увеличить количество моделей (для уменьшения первой компоненты формулы)
2) Ответы моделей должны быть как можно менее скоррелированы между собой (для уменьшения второй компоненты)

Исходя из поставленных требований в целях снижения смещения выбираем глубокие деревья, а в целях снижения ковариации (помимо ее снижения от бутстрепа) изменяем алгоритм работы дерева (каждый раз при выборе лучшего предиката ограничить набор признаков classification $m = \sqrt d$ regression $m = \frac{d}{3}$). Полученный алгоритм и является случайным лесом (Random Forest).

Одним из интересных свойств случайного леса является Out-of-bag оценка. Идея заклчается в том чтобы проводить оценку по объектам, которые не попали в обучающую выборку из-за бутсрепа.

Слабые стороны данного алгоритма скорость работы (длительный процесс построения большого количества глубоких деревьев).

### Реализация алгоритма на python

**DecisionTreesMyForRF** - класс с дополнетельным параметром m в fit для рандомизации количества признаков при поиске лучшего предиката


```python
#импорт необходимых библиотек
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import tree
from DecisionTreesMyForRF import DecisionTreeRegressorMy
import warnings
warnings.filterwarnings('ignore')
```


```python
#модель построения случайного леса
class RandomForestMy():
    #инициализация
    def __init__(self):
        pass
    
    #обучение состоит в создании n моделей дерева решений
    def fit(self, X, y, deep=30, n=100):
        self.trees = []
        for _ in range(n):
            model_my = DecisionTreeRegressorMy()
            sample = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=True)
            model_my.fit(X[sample], y[sample], m=6)
            self.trees.append(model_my)
            
    #выдача прогноза массиву входных векторов X
    def predict(self, X):
        return np.mean([i.predict(X) for i in self.trees], axis=0)
```


```python
# импортируем датасет
from sklearn.datasets import fetch_california_housing

# загружаем и делаем выборку в 2000
housing = fetch_california_housing()
sample = np.random.choice(range(housing['data'].shape[0]), size=2000, replace=True)
X = housing['data'][sample]
y = housing['target'][sample]

#сплит данных на трейн и тест
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)

#построение и прогноз модели решающего дерева
model_my = DecisionTreeRegressorMy()
model_my.fit(X_train, y_train, max_deep=5)
print(f'tree mse {mean_squared_error(y_test, model_my.predict(X_test))}')

#построение и прогноз случайного леса
modelRF = RandomForestMy()
modelRF.fit(X_train, y_train)
print(f'RF mse {mean_squared_error(y_test, modelRF.predict(X_test))}')
```

    tree mse 0.62401112894789
    RF mse 0.6035235893862753
    


#machine_learning #my_ml_algoritm 
