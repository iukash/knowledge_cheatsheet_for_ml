## Время для различных функций роста O(n)


```python
import math
t = [1000, 60000, 3600000, 86400000, 2592000000, 31104000000, 3110400000000]
lg_n = [math.log(n) for n in t]
sqrt_n = [math.sqrt(n) for n in t]
n = [n for n in t]
n_lg_n = [n*math.log(n) for n in t]
n2 = [n**2 for n in t]
n3 = [n**3 for n in t]
```


```python
import pandas as pd
table_On = pd.DataFrame([lg_n, sqrt_n, n, n_lg_n, n2, n3], 
                        index=['lg_n', 'sqrt_n', 'n', 'n_lg_n', 'n2', 'n3'], 
                        columns=['секунда', 'минута', 'час', 'день', 'месяч', 'год', 'век'])
table_On
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
      <th>секунда</th>
      <th>минута</th>
      <th>час</th>
      <th>день</th>
      <th>месяч</th>
      <th>год</th>
      <th>век</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lg_n</th>
      <td>6.907755e+00</td>
      <td>1.100210e+01</td>
      <td>15.096444</td>
      <td>18.274498</td>
      <td>21.675696</td>
      <td>24.160602</td>
      <td>28.765772</td>
    </tr>
    <tr>
      <th>sqrt_n</th>
      <td>3.162278e+01</td>
      <td>2.449490e+02</td>
      <td>1897.366596</td>
      <td>9295.160031</td>
      <td>50911.688245</td>
      <td>176363.26148</td>
      <td>1763632.614804</td>
    </tr>
    <tr>
      <th>n</th>
      <td>1.000000e+03</td>
      <td>6.000000e+04</td>
      <td>3600000</td>
      <td>86400000</td>
      <td>2592000000</td>
      <td>31104000000</td>
      <td>3110400000000</td>
    </tr>
    <tr>
      <th>n_lg_n</th>
      <td>6.907755e+03</td>
      <td>6.601260e+05</td>
      <td>54347199.852335</td>
      <td>1578916647.398098</td>
      <td>56183403035.21125</td>
      <td>751491372857.541016</td>
      <td>89473058632251.453125</td>
    </tr>
    <tr>
      <th>n2</th>
      <td>1.000000e+06</td>
      <td>3.600000e+09</td>
      <td>12960000000000</td>
      <td>7464960000000000</td>
      <td>6718464000000000000</td>
      <td>967458816000000000000</td>
      <td>9674588160000000000000000</td>
    </tr>
    <tr>
      <th>n3</th>
      <td>1.000000e+09</td>
      <td>2.160000e+14</td>
      <td>46656000000000000000</td>
      <td>644972544000000000000000</td>
      <td>17414258688000000000000000000</td>
      <td>30091839012864000000000000000000</td>
      <td>30091839012864000000000000000000000000</td>
    </tr>
  </tbody>
</table>
</div>




#algoritms
