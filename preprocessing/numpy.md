### [Numpy](https://numpy.org/doc/)  `import numpy as np`
* линейная алгебра
 * [dot()](https://numpy.org/doc/stable/reference/generated/numpy.dot.html) - скалярное произведение векторов/матриц `np.dot(vector1, vector2)`
 * [linalg.inv()](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html) - вычислить обратную матрицу `np.linalg.inv(matrix)`
* расчет расстояний
 * [distance.euclidean()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html) - евклидово расстояние между векторами
  * `import numpy as np from scipy.spatial import distance`
  * `distance.euclidean(a, b)`
 * [distance.cityblock()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cityblock.html) - Манхэттенское расстояние между векторами
  * `import numpy as np from scipy.spatial import distance`
  * `distance.cityblock(a, b)`
* другие полезные функции
 * [arange()](https://numpy.org/doc/stable/reference/generated/numpy.arange.html) - в отличии от range работает с дробными числами `np.arange(0, 0.3, 0.02)`
 * [argmin()](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html) - индекс минимального элемента `np.array(ar).argmin()`
 * [argmax()](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html) - индекс максимального элемента `np.array(ar).argmax()`
 * [var()](https://numpy.org/doc/stable/reference/generated/numpy.var.html) - расчет дисперсии `np.var(array)`
 * [std()](https://numpy.org/doc/stable/reference/generated/numpy.std.html) - расчет среднего квадратичного отклонения СКО `np.std(array)`

#machine_learning #preprocessing 