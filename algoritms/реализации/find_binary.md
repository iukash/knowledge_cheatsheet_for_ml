## Бинарный поиск


```python
def find_binary(nums, key):
    start_index = 0
    end_index = len(nums)
    while True:
        index_mid = start_index + int((end_index - start_index)/2)
        if key == nums[index_mid]:
            return index_mid
        elif key > nums[index_mid]:
            start_index = index_mid
        else:
            end_index = index_mid
    
```


```python
lst = list(range(1, 100))
find_binary(lst, 17)
```




    16


#algoritms