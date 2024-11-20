## Сортировка слиянием


```python
def merge(A, p, q, r):
    L = A[p:q]
    R = A[q:r]
    nl = len(L)
    nr = len(R)
    i = 0
    j = 0
    for k in range(p, r):
        if i >= nl:
            A[k] = R[j]
            j += 1
        elif j >= nr:
            A[k] = L[i]
            i += 1
        elif L[i] <= R[i]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1
```


```python
def merge_sort(A, p, r):
    if len(A[p:r]) >= 2:
        q = (p + r) // 2
        merge_sort(A, p, q)
        merge_sort(A, q, r)
        merge(A, p, q, r)
```


```python
nums = list(range(20))
nums.reverse()
print(nums)
merge_sort(nums, 0, 20)
print(nums)
```

    [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    


#algoritms
