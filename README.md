# Numpy & Pandas
Used for learned data processing

• Fast calculation speed: numpy and pandas are both written in C language, and pandas is based on numpy, which is an upgraded version of numpy.
• Less resource consumption: Matrix calculations are used, which will be much faster than the dictionary or list that comes with python

## Numpy Attributes
• ndim: dimension
• shape: number of rows and columns
• size: the number of elements

## Numpy - create array
• array: create an array
• dtype: specify the data type and format (int, float)
• zeros: create the value of data all 0
• ones: create the value of data all 1
• empty: create the value ofdata close to 0
• arrange: create data in a specified range
• linspace: create line segments


## Several basic operations of Numpy (1)-element calculation and search operations
### One-dimensional：
(1)-/+/*
code:
import numpy as np
a=np.array([10,20,30,40])   # array([10, 20, 30, 40])
b=np.arange(4)              # array([0, 1, 2, 3])

operation     #output
c=a-b         #array([10, 19, 28, 37])
c=a+b         #array([10, 21, 32, 43])
c=a*b         #array([  0,  20,  60, 120])

(2)Quadratic:
   c=b**2  #array([0, 1, 4, 9])

(3)c=10*np.sin(a)  
   #array([-5.44021111,  9.12945251, -9.88031624,  7.4511316 ])

(4)boolean: 
         print(b<3)  
         #array([ True,  True,  True, False], dtype=bool)


### Multi-row and multi-dimensional matrix：
(1) *
a=np.array([[1,1],[0,1]])
b=np.arange(4).reshape((2,2))

print(a)
#array([[1, 1],
       [0, 1]])

print(b)
#array([[0, 1],
       [2, 3]])

c_dot = np.dot(a,b)
#array([[2, 4],
       [2, 3]])
other way:
c_dot_2 = a.dot(b)
#array([[2, 4],
       [2, 3]])

(2) sum(), min(), max()的使用：
import numpy as np
a=np.random.random((2,4))
print(a)
#array([[ 0.94692159,  0.20821798,  0.35339414,  0.2805278 ],
       [ 0.04836775,  0.04023552,  0.44091941,  0.21665268]])

np.sum(a)   #4.4043622002745959
np.min(a)   #0.23651223533671784
np.max(a)   #0.90438450240606416

(3) Operation by rows or columns - axis 

print("a =",a)
#a = [[ 0.23651224  0.41900661  0.84869417  0.46456022]
#[ 0.60771087  0.9043845   0.36603285  0.55746074]]

print("sum =",np.sum(a,axis=1))
#sum = [ 1.96877324  2.43558896]

print("min =",np.min(a,axis=0))
#min = [ 0.23651224  0.41900661  0.36603285  0.46456022]

print("max =",np.max(a,axis=1))
#max = [ 0.84869417  0.9043845 ]


## Several basic operations of Numpy (2)- The index of the corresponding element

import numpy as np
A = np.arange(2,14).reshape((3,4)) 

#array([[ 2, 3, 4, 5]
        [ 6, 7, 8, 9]
        [10,11,12,13]])

(1)
print(np.argmin(A))    # 0      
print(np.argmax(A))    # 11 

print(np.mean(A))        # 7.5
print(np.average(A))     # 7.5

print(A.mean())          # 7.5

print(A.median())       # 7.5

(2)
Accumulation function：print(np.cumsum(A))  # [2 5 9 14 20 27 35 44 54 65 77 90]

print(np.diff(A))    
#[[1 1 1]
  [1 1 1]
  [1 1 1]]

(3)
print(np.nonzero(A))    

#(array([0,0,0,0,1,1,1,1,2,2,2,2]),array([0,1,2,3,0,1,2,3,0,1,2,3]))

(4)Sorting operation: But the sorting function here still only sorts from small to large for each row:

import numpy as np
A = np.arange(14,2, -1).reshape((3,4)) 

#array([[14, 13, 12, 11],
       [10,  9,  8,  7],
       [ 6,  5,  4,  3]])

print(np.sort(A))    

#array([[11,12,13,14]
        [ 7, 8, 9,10]
        [ 3, 4, 5, 6]])

(5)transposition: 
print(np.transpose(A))    
print(A.T)

#array([[14,10, 6]
        [13, 9, 5]
        [12, 8, 4]
        [11, 7, 3]])
#array([[14,10, 6]
        [13, 9, 5]
        [12, 8, 4]
        [11, 7, 3]])

(6)Clip(Array,Array_min,Array_max)
print(A)
#array([[14,13,12,11]
        [10, 9, 8, 7]
        [ 6, 5, 4, 3]])

print(np.clip(A,5,9))    
#array([[ 9, 9, 9, 9]
        [ 9, 9, 8, 7]
        [ 6, 5, 5, 5]])


## Numpy 索引
### One-dimensional
import numpy as np
A = np.arange(3,15)

#array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
         
print(A[3])    # 6

2D:
A = np.arange(3,15).reshape((3,4))
"""
array([[ 3,  4,  5,  6]
       [ 7,  8,  9, 10]
       [11, 12, 13, 14]])
"""
         
print(A[2])         
#[11 12 13 14]


### Two-dimensional index

print(A[1][1])      # 8
print(A[1, 1])      # 8

print(A[1, 1:3])    # [8 9]
print(A[2, :])    # [11 12 13 14]
print(A[:, 1])    # [4  8 12]

for row in A:
    print(row)
"""    
[ 3,  4,  5, 6]
[ 7,  8,  9, 10]
[11, 12, 13, 14]
"""

for column in A.T:
    print(column)
"""  
[ 3,  7,  11]
[ 4,  8,  12]
[ 5,  9,  13]
[ 6, 10,  14]
"""

import numpy as np
A = np.arange(3,15).reshape((3,4))
         
print(A.flatten())   
#array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

for item in A.flat:
    print(item)
    
#3
#4
……
#14


## Numpy array 合并
### np.vstack() 

import numpy as np
A = np.array([1,1,1])
B = np.array([2,2,2])
         
print(np.vstack((A,B)))    # vertical stack
"""
[[1,1,1]
 [2,2,2]]
"""

C = np.vstack((A,B))      
print(A.shape,C.shape)

#(3,) (2,3)

### np.hstack() 

D = np.hstack((A,B))       # horizontal stack

print(D)
#[1,1,1,2,2,2]

print(A.shape,D.shape)
#(3,) (6,)

### np.newaxis() 
print(A[np.newaxis,:])
#[[1 1 1]]

print(A[np.newaxis,:].shape)
#(1,3)

print(A[:,np.newaxis])
"""
[[1]
[1]
[1]]
"""

print(A[:,np.newaxis].shape)
#(3,1)


import numpy as np
A = np.array([1,1,1])[:,np.newaxis]
B = np.array([2,2,2])[:,np.newaxis]
         
C = np.vstack((A,B))   # vertical stack
D = np.hstack((A,B))   # horizontal stack

print(D)
"""
[[1 2]
[1 2]
[1 2]]
"""

print(A.shape,D.shape)
#(3,1) (3,2)

### np.concatenate() 

C = np.concatenate((A,B,B,A),axis=0)

print(C)
"""
array([[1],
       [1],
       [1],
       [2],
       [2],
       [2],
       [2],
       [2],
       [2],
       [1],
       [1],
       [1]])
"""

D = np.concatenate((A,B,B,A),axis=1)

print(D)
"""
array([[1, 2, 2, 1],
       [1, 2, 2, 1],
       [1, 2, 2, 1]])
"""


## Numpy array 分割

import numpy as np
A = np.arange(12).reshape((3, 4))
print(A)
"""
array([[ 0,  1,  2,  3],
    [ 4,  5,  6,  7],
    [ 8,  9, 10, 11]])
"""
### Vertical split
print(np.split(A, 2, axis=1))
"""
[array([[0, 1],
        [4, 5],
        [8, 9]]), 
 array([[ 2,  3],
        [ 6,  7],
        [10, 11]])]
"""
### Horizontal split
print(np.split(A, 3, axis=0))

#[array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]

print(np.split(A, 3, axis=1))
#ValueError: array split does not result in an equal division

### Unequal division
print(np.array_split(A, 3, axis=1))
"""
[array([[0, 1],
        [4, 5],
        [8, 9]]), 
 array([[ 2],
        [ 6],
        [10]]), 
 array([[ 3],
        [ 7],
        [11]])]
"""

### other
(1)print(np.vsplit(A, 3)) 

#[array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]


(2)print(np.hsplit(A, 2)) 
"""
[array([[0, 1],
       [4, 5],
       [8, 9]]), 
array([[ 2,  3],
        [ 6,  7],
        [10, 11]])]
"""


## Numpy copy & deep copy
### =
import numpy as np

a = np.arange(4)
#array([0, 1, 2, 3])

b = a
c = a
d = b

a[0] = 11
print(a)
#array([11,  1,  2,  3])

b is a  # True
c is a  # True
d is a  # True

d[1:3] = [22, 33]   # array([11, 22, 33,  3])
print(a)            # array([11, 22, 33,  3])
print(b)            # array([11, 22, 33,  3])
print(c)            # array([11, 22, 33,  3])

### copy()
b = a.copy()    # deep copy
print(b)        # array([11, 22, 33,  3])
a[3] = 44
print(a)        # array([11, 22, 33, 44])
print(b)        # array([11, 22, 33,  3])
此时a与b已经没有关联。














