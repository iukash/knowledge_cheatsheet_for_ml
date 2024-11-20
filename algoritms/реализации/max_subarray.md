## Поиск максимального подмассива

### Задача поиска максимальной выгоды при купле/продаже акций на определенном отрезке времени

Создадим рандомный массив (фиксированный), визуально проанализируем, отметим максимальный подмассив.


```python
import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(1000)
price = np.random.normal(loc=0, scale=50, size=100)
price += 150

plt.figure(figsize=(17, 5))
min_price = min(price)
max_price = max(price)
plt.vlines(x=60, ymin=min_price, ymax=max_price, colors='green', ls='--', lw=2)
plt.vlines(x=91, ymin=min_price, ymax=max_price, colors='green', ls='--', lw=2)
plt.fill_between(x=[60, 91], y1=min_price, y2=max_price, color='lightgreen')
plt.plot(price)
plt.text(72, 30, round(price[91] - price[60], 1), fontsize = 24)
plt.show()
```


    
![[max_subarray_1.png]]]
    


### Тривиальный алгоритм за O($n^2$)


```python
def find_max_subarray_trivial(price):
    maximum = 0
    left_ind = 0
    right_ind = 0

    for i in range(len(price)):
        for j in range(i, len(price)):
            value = round(price[j] - price[i], 1)
            if value > maximum:
                maximum = value
                left_ind = i
                right_ind = j
                
    return left_ind, right_ind, round(maximum, 1)
```


```python
left_ind, right_ind, maximum = find_max_subarray_trivial(price)
print(f'maximum = {maximum}, left_ind = {left_ind}, right_ind = {right_ind}')
```

    maximum = 249.5, left_ind = 60, right_ind = 91
    

### Алгориитм разделяй и властвуй за O(n*log n)

Задачe можно свести к нахождению максимального подмассива, если создать массив разниц текущего и предыдущего значения.


```python
import pandas as pd
price_dif = list(price[i+1] - price[i] for i in range(0, len(price)-1))
price_dif.insert(0, 0)
pd.DataFrame([price, price_dif], index=['price', 'difference'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>90</th>
      <th>91</th>
      <th>92</th>
      <th>93</th>
      <th>94</th>
      <th>95</th>
      <th>96</th>
      <th>97</th>
      <th>98</th>
      <th>99</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>price</th>
      <td>109.777085</td>
      <td>166.046577</td>
      <td>148.725856</td>
      <td>182.216191</td>
      <td>134.960166</td>
      <td>169.473728</td>
      <td>144.628135</td>
      <td>126.000846</td>
      <td>179.751775</td>
      <td>126.766624</td>
      <td>...</td>
      <td>89.864834</td>
      <td>293.088459</td>
      <td>15.920378</td>
      <td>91.559412</td>
      <td>120.791789</td>
      <td>190.912485</td>
      <td>229.521862</td>
      <td>137.085588</td>
      <td>172.351452</td>
      <td>246.920987</td>
    </tr>
    <tr>
      <th>difference</th>
      <td>0.000000</td>
      <td>56.269493</td>
      <td>-17.320721</td>
      <td>33.490335</td>
      <td>-47.256025</td>
      <td>34.513561</td>
      <td>-24.845593</td>
      <td>-18.627289</td>
      <td>53.750929</td>
      <td>-52.985151</td>
      <td>...</td>
      <td>-103.609931</td>
      <td>203.223625</td>
      <td>-277.168081</td>
      <td>75.639034</td>
      <td>29.232377</td>
      <td>70.120696</td>
      <td>38.609377</td>
      <td>-92.436273</td>
      <td>35.265864</td>
      <td>74.569535</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 100 columns</p>
</div>



Из таблицы очевидно, что после решения задачи нахождения максимального подмассива, из первого элемента необходимо вычесть 1 поскольку максимизируемая разница получается с его использованием.


```python
def find_mid(A, low, mid, high):
    left_sum = A[mid-1]
    left_index = mid-1
    sum = 0
    for i in range(mid-1, low-1, -1):
        sum += A[i]
        if sum > left_sum:
            left_sum = sum
            left_index = i
        
    right_sum = A[mid]
    right_index = mid
    sum = 0
    for i in range(mid, high, 1):
        sum += A[i]
        if sum > right_sum:
            right_sum = sum
            right_index = i
    return (left_index, right_index, left_sum + right_sum)

def find_max_subarray(A, low, high):
    if (high-1) == low:
        return (low, high, A[low])
    else:
        mid = int((high + low)/2)
        (left_low, left_high, left_sum) = find_max_subarray(A, low, mid)
        (right_low, right_high, right_sum) = find_max_subarray(A, mid, high)
        (cross_low, cross_high, cross_sum) = find_mid(A, low, mid, high)

        if left_sum > right_sum and left_sum > cross_sum:
            return (left_low, left_high, left_sum)
        elif right_sum >= left_sum and right_sum > cross_sum:
            return (right_low, right_high, right_sum)
        else:
            return (cross_low, cross_high, cross_sum)
```


```python
left_ind, right_ind, maximum = find_max_subarray(price_dif, 0, len(price_dif))
print(f'maximum = {round(maximum, 1)}, left_ind = {left_ind-1}, right_ind = {right_ind}')
```

    maximum = 249.5, left_ind = 60, right_ind = 91
    

Создадим функцию - добавив перевод массива в нужный формат и вычитание единицы из индекса левой границы


```python
def find_max_subarray_divide(price):
    price_dif = list(price[i+1] - price[i] for i in range(0, len(price)-1))
    price_dif.insert(0, 0)
    left_ind, right_ind, maximum = find_max_subarray(price_dif, 0, len(price_dif))
    return left_ind-1, right_ind, round(maximum, 1)
```


```python
left_ind, right_ind, maximum = find_max_subarray_divide(price)
print(f'maximum = {maximum}, left_ind = {left_ind}, right_ind = {right_ind}')
```

    maximum = 249.5, left_ind = 60, right_ind = 91
    

### Алгоритм поиска за линейное время O(n)


```python
def find_max_subarray_line(price):
    price_dif = list(price[i+1] - price[i] for i in range(0, len(price)-1))
    price_dif.insert(0, 0)
    left_ind = 0
    right_ind = 0
    prev = price_dif[0]
    prev_ind = 0
    maximum = price_dif[0]

    for i in range(1, len(price_dif)):
        if prev + price_dif[i] > price_dif[i]:
            prev += price_dif[i]
        else:
            prev = price_dif[i]
            prev_ind = i

        if prev > maximum:
            maximum = prev
            right_ind = i
            left_ind = prev_ind

    return left_ind-1, right_ind, round(maximum, 1)
```


```python
left_ind, right_ind, maximum = find_max_subarray_line(price)
print(f'maximum = {maximum}, left_ind = {left_ind}, right_ind = {right_ind}')
```

    maximum = 249.5, left_ind = 60, right_ind = 91
    

### Визуальное сравнение скорости работы алгоритмов


```python
def count_time(func, price):
    start_time = time.time()
    func(price)
    return(time.time() - start_time)
    
lst_n = []
lst_time_trivial = []
lst_time_divide = []
lst_time_line = []
n = 1

for x in range(1, 12, 2):
    n = 2**x
    lst_n.append(n)
    price = np.random.normal(loc=0, scale=50, size=n)
    price += 150
    lst_time_trivial.append(count_time(find_max_subarray_trivial, price))
    lst_time_divide.append(count_time(find_max_subarray_divide, price))
    lst_time_line.append(count_time(find_max_subarray_line, price))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_7320\1521054668.py in <module>
         13     n = 2**x
         14     lst_n.append(n)
    ---> 15     price = np.random.normal(loc=0, scale=50, size=n)
         16     price += 150
         17     lst_time_trivial.append(count_time(find_max_subarray_trivial, price))
    

    NameError: name 'np' is not defined



```python
plt.figure(figsize=(17, 5))
plt.plot(lst_n, lst_time_trivial, label='n2', color='blue')
plt.plot(lst_n, lst_time_divide, label='nlogn', color='green')
plt.plot(lst_n, lst_time_line, label='n', color='orange')
plt.legend()
plt.show()
```


    
![[max_subarray_2.png]]]
    



```python
plt.figure(figsize=(17, 5))
plt.plot(lst_n, lst_time_divide, label='nlogn', color='green')
plt.plot(lst_n, lst_time_line, label='n', color='orange')
plt.legend()
plt.show()
```


    
![[max_subarray_3.png]]
    



#algoritms
