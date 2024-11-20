```python
import random
import pandas as pd
import matplotlib.pyplot as plt

def kubik(N):
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0
    six = 0
    for i in range(0, N):
        rand = random.randint(1,6)
        if rand == 1:
            one += 1
        elif rand == 2:
            two += 1
        elif rand == 3:
            three += 1
        elif rand == 4:
            four += 1
        elif rand == 5:
            five += 1
        elif rand == 6:
            six += 1
    return [one/N, two/N, three/N, four/N, five/N, six/N]

vib = kubik(1000000)
plt.bar([1, 2, 3, 4, 5, 6], vib)
```


    
![[law_big_numeric_kubik.png]]

#machine_learning #mathematica 