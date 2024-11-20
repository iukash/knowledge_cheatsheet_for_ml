## Разбор метода иерархий Саати

### Описание метода

Итак, опишем метод анализа иерархий Т.Саати. Собственно он предлагает экспертам попарно оценивать значимость одного признака/свойства объекта над другим в конечном итоге формируя обратносимметричную матрицу иерархий и дальше магическим образом через собственные вектора и значения и непонятные возведения матрицы в произвольно большие степени (короче говоря "вжух") получать вектор весов (значимость) каждого принципа. Процесс формирования матрицы не сложен и описан множество раз, поэтому сконценрируемся на математической подкапотной "вжух" части этого метода.

Введем необходимые понятия: 
* матрица A - обратносимметричная (кстати этого определения нигде кроме контекста Саати и нет) матрица  парных сравнений
* w - вектор весов (значимости) признаков
* согласованность - свойство матрица при котором $a_{ik} =a_{ij}a_{jk}$ (если объект А больше объекта В в 2 раза, а объект В больше объекта С в 3 раза, объект А должен быть больше объекта С в 6 раз)

Продемонстрируем пример согласованной матрицы для случая когда зависимость известна заранее (т.е. мы знаем веса), например ценность какого либо объекта зависит от его характеристик длины (l), ширины (d) и высоты (h) так, что длина важнее ширины в 2 раза, ширина важнее высоты в 3 раза , соответственно длина важнее высоты в 6 раз.
Тогда матрица парных сравнений примет вид:

$$
  A =
  \left[ {\begin{array}{cc}
    1 & 2 & 6 \\
    1/2 & 1 & 3 \\
    1/6 & 1/3 & 1 \\
  \end{array} } \right]
$$

Также из формулировки мы можем опредлить веса и конечную формулу расчета ценности
$G = l + \frac{1}{2}*d + \frac{1}{6}*h$, откуда соответственно видно, что веса равны $w_1=1$ $w_2=\frac{1}{2}$ $w_3=\frac{1}{6}$

Или после нормировки равны


```python
import numpy as np
w = np.array([1, 1/2, 1/6])
print(w/np.sum(w))
```

    [0.6 0.3 0.1]
    

Очевидно, что при известных весах признаков можно опредлить элементы матрицы по формуле $a_{ij}=\frac{w_i}{w_j}$, откуда получаем
$$a_{ij} \frac{w_j}{w_i}=1, \text{ i,j = 1, 2, ..., n}$$
и следовательно
$$\sum_{j=1}^n a_{ij} w_j \frac{1}{w_i} = n, \text{ i=1, 2, ..., n}$$
или
$$\sum_{j=1}^n a_{ij} w_j =n w_i, \text{ i=1, 2, ..., n}$$
что эквивалентно выражению
$$Aw=nw$$

Выполним проверку изложенной теории на определенном выше примере.


```python
#определим матрицу А
A = np.array([[1, 2, 6], [1/2, 1, 3], [1/6, 1/3, 1]])
#рассчитаем собственные вектора и значения
L, W = np.linalg.eig(A)
print(f'собственные значения матрицы А равны {np.round(L, 2)}')
```

    собственные значения матрицы А равны [-0.  3.  0.]
    

А значения соответствующего максимальному собственному значению собственного верктора равны:


```python
print(W.T[1])
print('или после нормировки')
print(W.T[1]/np.sum(W.T[1]))
```

    [0.88465174 0.44232587 0.14744196]
    или после нормировки
    [0.6 0.3 0.1]
    

Из проверки видим, что при согласованной матрице мы имеем, что максимальное собственное значение примерно равно размерности матрицы, а соответствующий собственный вектор - заданному вектору весов.

Основываясь на факте, что при незначительном изменении матрицы A (матрица перестает быть согласованной, кстати термин согласованной в отношении матриц тоже походу придумал Саати) собственные значения также изменяются незначительно автор считает возможным представить формулу в следующем виде
$$A*w=\lambda_{max}*w$$
*Справочно: в принципе логично, если эксперт точно знает что описывает и при этом полученная матрица близка к согласованной оценки важность признаков должна быть достаточно точна*

**Важно!** Однако, если в силу сложности описываемого экспертом процесса, матрица получается плохо согласованна, то значения собственного значения не будут правильно описывать истинную зависимость (важность признаков).

В связи с этим автор вводит такое понятие как согласованность, которая рассчитывается по следующей формуле
$$ИС = \frac{\lambda_{max} - n}{n-1}$$
И предлагает оценивать согласованность суждений эксперта положительно, если этот показатель меньше или равен 0,1 (хотя цифру 0,1 автор взял с потолка и никак не обосновывает, ну а хуля, мы тоже в своих отчетах так делаем). В целом логика есть в том, что чем ниже этот индекс, тем согласованней суждения эксперта, и дает нам возможность сформировать подход к **обобщению мнения нескольких экспертов** используя некий порог индекса согласованности (например предложенный автором 0,1) и смело выбрасывать экспертов с индексом превыщающим этот порог (ну или возвращать им их формулировки по попарной оценке важности со словами "говно, переделать!").

**P.S. Оценивать индекс согласованности супер ВАЖНО!**

### Разбор математики по нахождению максимального собственного значения используя степенной метод

Ну для начала наверное надо понять зачем он нам вообще нужен этот степенной метод и можно ли обойтись без него. Рассмотрим стандартный способ поиска собственных значений через характеристическое уравнение. Продолжим мучать нашу матрицу А (надеюсь я смогу решить уравнение 3-го порядка и там будут действительные корни, иначе придется показывать на матрице 2x2).

$$
  A =
  \left[ {\begin{array}{cc}
    1 & 2 & 6 \\
    1/2 & 1 & 3 \\
    1/6 & 1/3 & 1 \\
  \end{array} } \right]
$$

Запишем формулу собственных векторов и значений
$$A*x=\lambda*x$$

Или если расписать

$$
  \left[ {\begin{array}{cc}
    1 & 2 & 6 \\
    1/2 & 1 & 3 \\
    1/6 & 1/3 & 1 \\
  \end{array} } \right]
    \left[ {\begin{array}{cc}
    x_1\\
    x_2\\
    x_3\\
  \end{array} } \right]
  =
  \lambda
      \left[ {\begin{array}{cc}
    x_1\\
    x_2\\
    x_3\\
  \end{array} } \right]
$$

выполним скалярное умножение левой и правой части

$$
\Bigl\{ {\begin{array}{cc}
    1*x_1 + 2*x_2 + 6*x_3 = \lambda*x1\\
    1/2*x_1 + 1*x_2 + 3*x_3 = \lambda*x2\\
    1/6*x_1 + 1/3*x_2 + 1*x_3 = \lambda*x3\\
  \end{array} }
$$

правую часть перенесем влево и чуть преобразуем

$$
\Bigl\{ {\begin{array}{cc}
    x_1*(1-\lambda) + 2*x_2 + 6*x_3 = 0\\
    1/2*x_1 + x_2*(1 - \lambda) + 3*x_3 = 0\\
    1/6*x_1 + 1/3*x_2 + x_3*(1 - \lambda) = 0\\
  \end{array} }
$$

Получили однородную систему линейных уравнений, которая имеет ненулевое решение только если ее дискриминант равен нулю (теорема Крамера). Приравняем дискриминант к нулю.

$$
  \left| {\begin{array}{cc}
    1-\lambda & 2 & 6 \\
    1/2 & 1-\lambda & 3 \\
    1/6 & 1/3 & 1-\lambda \\
  \end{array} } \right|
  = 0 \text{ - характеристическое уравнение матрицы А корни которого собственные числа}
$$

Раскроем дискриминант по правилу треугольников (главная диагональ + треугольники с основаниями параллельными главной диагонали - побочная диагональ и - треугольники основание которых параллельно ей).

$$(1-\lambda)*(1-\lambda)*(1-\lambda)+2*3/6+1/2*1/3*6-6*(1-\lambda)*1/6-1/2*2*(1-\lambda)-3*1/3*(1-\lambda) = 0$$

Преобразования расписывать не буду, если аккуратно раскрыть скобки и поплюсовать получим

$$\lambda^2*(\lambda - 3)=0$$

Откуда очевидна, что имеем один корень $\lambda = 3$ (я забыл про согласованность этой матрицы и единственное собственное значение, так бы было не сладко). 

И еще вывод который можно сделать, что степень характеристического уравнения равна n, а это значит что уже нахождение корней для n=3 может быть не простой задачей, не говоря уже о более высоких порядках.

**Степенной метод** позволяет найти максимальное собственное значение и соответствующий ему собственный вектор. Распишу алгоритм его применения параллельно показав почему он работает.
1) Пронумеруем собственные значения в порядке возрастания $\lambda_1>\lambda_2>...>\lambda_n$
2) Возьмем произвольный вектор $u_0$ размером n (строго говоря нам с ним должно повезти чтобы он не оказался ортогонален собственному подпространству и есть способы борьбы с этим, но мы будем верить в удачу, да и вероятность этого как мне кажется достаточно мала).
3) Представим вектор $u_0$ в виде линейной комбинации собственных векторов (несмотря на то что они нам пока неизвестны)
$$u_0 = c_1*x_1+c_2*x_2+...+c_n*x_n$$
4) Вычислим вектор u1 по формуле
$$u_1 = A*u_0 = |\text{в силу свойства }A*x=\lambda*x|=c_1*\lambda_1*x_1+c_2*\lambda_2*x_2+...+c_n*\lambda_n*x_n$$
P.S. строго говоря
6) Вычислим $u_2 = c_1*\lambda_1^2*x_1+c_2*\lambda_2^2*x_2+...+c_n*\lambda_n^2*x_n$
7) Соответственно $u_k = c_1*\lambda_1^k*x_1+c_2*\lambda_2^k*x_2+...+c_n*\lambda_n^k*x_n$
8) Найдем отношение вектора $u_{k+1}$ на вектор $u_{k}$ покомпанентно, тогда для ненулевой компоненты s
$$\frac{us_{k+1}}{us_k} = \frac{c_1*\lambda_1^{k+1}*x_1+c_2*\lambda_2^{k+1}*x_2+...+c_n*\lambda_n^{k+1}*x_n}{c_1*\lambda_1^k*x_1+c_2*\lambda_2^k*x_2+...+c_n*\lambda_n^k*x_n}$$
в числителе и знаменателе вынесем за скобки первое слагаемое, получим
$$\frac{us_{k+1}}{us_k} = \frac{c_1*\lambda_1^{k+1}*x_1}{c_1*\lambda_1^k*x_1}*\frac{1 + \frac{c_2*\lambda_2^{k+1}*x_2}{c_1*\lambda_1^{k+1}*x_1}+...+\frac{c_n*\lambda_n^{k+1}*x_n}{c_1*\lambda_1^{k+1}*x_1}}{1 + \frac{c_2*\lambda_2^k*x_2}{c_1*\lambda_1^k*x_1}+...+\frac{c_n*\lambda_n^k*x_n}{c_1*\lambda_1^k*x_1}} = \lambda_1*\frac{1 + \frac{c_2*\lambda_2^{k+1}*x_2}{c_1*\lambda_1^{k+1}*x_1}+...+\frac{c_n*\lambda_n^{k+1}*x_n}{c_1*\lambda_1^{k+1}*x_1}}{1 + \frac{c_2*\lambda_2^k*x_2}{c_1*\lambda_1^k*x_1}+...+\frac{c_n*\lambda_n^k*x_n}{c_1*\lambda_1^k*x_1}}$$
Рассмотрев второй множитель во всех слагаемых кроме единицы видим одинаковое отношение (с разными степенями k и k+1)
$$\frac{\lambda_j^k}{\lambda_1^k}=(\frac{\lambda_j}{\lambda_1})^k \text{ поскольку } |\frac{\lambda_j}{\lambda_1}| < 1 \text{ тогда } (\frac{\lambda_j}{\lambda_1})^k \rightarrow 0 \text{ при k стремящемся к }\infty$$
и следовательно, поскольку второй множитель стремится к нулю при k стремящемся к бесконечности, тогда
$$\frac{us_{k+1}}{us_k} \rightarrow \lambda_1 \text{ при k стремящемся к }\infty$$
9) В качестве критерия сходимости можно определить
$$|\frac{us_{k+1}}{us_k} - \frac{us_{k}}{us_{k-1}}| < \epsilon$$
10) Покажем, что $u_{k+1}$ является собственным вектором
$$u_{k+1} = c_1*\lambda_1^{k+1}*x_1+c_2*\lambda_2^{k+1}*x_2+...+c_n*\lambda_n^{k+1}*x_n = c_1*\lambda_1^{k+1}*x_1*(1 + \frac{c_2*\lambda_2^{k+1}*x_2}{c_1*\lambda_1^{k+1}*x_1}+...+\frac{c_n*\lambda_n^{k+1}*x_n}{c_1*\lambda_1^{k+1}*x_1}) \rightarrow c_1*\lambda_1^{k+1}*x_1 = \nabla*x_1$$

### Реализация на python


```python
#определим матрицу А
A = np.array([[1, 2, 6], [1/2, 1, 3], [1/6, 1/3, 1]])
#зададим случайный вектор u0
u0 = np.array([1, 1, 1])
#ну поскольку мы програмируем создадим функцию расчета и зациклим ее до срабатывания критерия останова/сходимости
def calculate_next_u(A, u):
    return np.dot(A, u)

#сделаем несколько шагов для расчета критерия сходимости
u_cur = np.copy(u0)
u_next = np.copy(calculate_next_u(A, u_cur))
u_next2 = np.copy(calculate_next_u(A, u_next))
#количество шагов
i = 2

#будем возводить в степень до потери пульса (до сходимости с заданной точностью)
while u_next2[0]/u_next[0] - u_next[0]/u_cur[0] > 0.01:
    u_cur = np.copy(u_next)
    u_next = np.copy(u_next2)
    u_next2 = np.copy(calculate_next_u(A, u_next))
    i+=1

print(f'конечная степень при вычислении {i}')
print(f'нормированный собственный вектор {u_next2/np.sum(u_next2)}')
print(f'полученное максимальное собственное значение {(u_next2[0])/(u_next[0])}')
print(f'максимальное собственное значение при расчете используя библиотеку numpy {np.round(np.linalg.eig(A)[0].max(), 2)}')
```

    конечная степень при вычислении 2
    нормированный собственный вектор [0.6 0.3 0.1]
    полученное максимальное собственное значение 3.0
    максимальное собственное значение при расчете используя библиотеку numpy 3.0
    


#machine_learning #mathematica