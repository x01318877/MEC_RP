# Numpy & Pandas
Used for learned data processing

• Fast calculation speed: numpy and pandas are both written in C language, and pandas is based on numpy, which is an upgraded version of numpy.
• Less resource consumption: Matrix calculations are used, which will be much faster than the dictionary or list that comes with python

## Numpy 属性
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

code:
import numpy as np
a=np.array([10,20,30,40])   # array([10, 20, 30, 40])
b=np.arange(4)              # array([0, 1, 2, 3])

operation     #output
c=a-b         #array([10, 19, 28, 37])
c=a+b         #array([10, 21, 32, 43])
c=a*b         #array([  0,  20,  60, 120])

在Numpy中，想要求出矩阵中各个元素的乘方需要依赖双星符号 **，以二次方举例，即：
c=b**2  # array([0, 1, 4, 9])
数学函数工具，比如三角函数等，对矩阵中每一项元素进行函数运算时（以sin函数为例）：
c=10*np.sin(a)  
# array([-5.44021111,  9.12945251, -9.88031624,  7.4511316 ])
除了函数应用外，在脚本中对print函数进行一些修改可以进行逻辑判断：
print(b<3)  
# array([ True,  True,  True, False], dtype=bool)
此时由于进行逻辑判断，返回的是一个bool类型的矩阵，即对满足要求的返回True，不满足的返回False。如果想要执行是否相等的判断， 依然需要输入 == 而不是 = 来完成相应的逻辑判断。

对多行多维度的矩阵进行操作（2行2列）：
a=np.array([[1,1],[0,1]])
b=np.arange(4).reshape((2,2))

print(a)
# array([[1, 1],
#       [0, 1]])

print(b)
# array([[0, 1],
#       [2, 3]])
Numpy中的矩阵乘法分为两种， 其一是前文中的对应元素相乘，其二是标准的矩阵乘法运算，即对应行乘对应列得到相应元素：
c_dot = np.dot(a,b)
# array([[2, 4],
#       [2, 3]])
除此之外还有另外的一种关于dot的表示方法，即：
c_dot_2 = a.dot(b)
# array([[2, 4],
#       [2, 3]])
矩阵怎样计算: https://www.jianshu.com/p/09f4174a723f
sum(), min(), max()的使用：
import numpy as np
a=np.random.random((2,4))
print(a)
# array([[ 0.94692159,  0.20821798,  0.35339414,  0.2805278 ],
#       [ 0.04836775,  0.04023552,  0.44091941,  0.21665268]])
因为是随机生成数字, 所以你的结果可能会不一样. 在第二行中对a的操作是令a中生成一个2行4列的矩阵，且每一元素均是来自从0到1的随机数。 在这个随机生成的矩阵中，我们可以对元素进行求和以及寻找极值的操作，具体如下：
np.sum(a)   # 4.4043622002745959
np.min(a)   # 0.23651223533671784
np.max(a)   # 0.90438450240606416

如果你需要对行或者列进行查找运算，就需要在上述代码中为 axis 进行赋值。 当axis的值为0的时候，将会以列作为查找单元， 当axis的值为1的时候，将会以行作为查找单元。
为了更加清晰，在刚才的例子中我们继续进行查找：
print("a =",a)
# a = [[ 0.23651224  0.41900661  0.84869417  0.46456022]
# [ 0.60771087  0.9043845   0.36603285  0.55746074]]

print("sum =",np.sum(a,axis=1))
# sum = [ 1.96877324  2.43558896]

print("min =",np.min(a,axis=0))
# min = [ 0.23651224  0.41900661  0.36603285  0.46456022]

print("max =",np.max(a,axis=1))
# max = [ 0.84869417  0.9043845 ]



Numpy 的几种基本运算 （2）-- 对应元素的索引
import numpy as np
A = np.arange(2,14).reshape((3,4)) 

# array([[ 2, 3, 4, 5]
#        [ 6, 7, 8, 9]
#        [10,11,12,13]])
         
print(np.argmin(A))    # 0      最小元素‘2’的索引
print(np.argmax(A))    # 11      最大元素‘13’的索引
如果需要计算统计中的均值，可以利用下面的方式，将整个矩阵的均值求出来：
print(np.mean(A))        # 7.5
print(np.average(A))     # 7.5
仿照着前一节中dot() 的使用法则，mean()函数还有另外一种写法：
print(A.mean())          # 7.5
中位数的函数：print(A.median())       # 7.5
累加函数：print(np.cumsum(A))  # [2 5 9 14 20 27 35 44 54 65 77 90]
累差运算函数：该函数计算的便是每一行中后一项与前一项之差。故一个3行4列矩阵通过函数计算得到的矩阵便是3行3列的矩阵。
print(np.diff(A))    
# [[1 1 1]
#  [1 1 1]
#  [1 1 1]]
nonzero()函数：
print(np.nonzero(A))    

# (array([0,0,0,0,1,1,1,1,2,2,2,2]),array([0,1,2,3,0,1,2,3,0,1,2,3]))
这个函数将所有非零元素的行与列坐标分割开，重构成两个分别关于行和列的矩阵。
排序操作：但这里的排序函数仍然仅针对每一行进行从小到大排序操作：
import numpy as np
A = np.arange(14,2, -1).reshape((3,4)) 

# array([[14, 13, 12, 11],
#       [10,  9,  8,  7],
#       [ 6,  5,  4,  3]])

print(np.sort(A))    

# array([[11,12,13,14]
#        [ 7, 8, 9,10]
#        [ 3, 4, 5, 6]])
矩阵的转置有两种表示方法：
print(np.transpose(A))    
print(A.T)

# array([[14,10, 6]
#        [13, 9, 5]
#        [12, 8, 4]
#        [11, 7, 3]])
# array([[14,10, 6]
#        [13, 9, 5]
#        [12, 8, 4]
#        [11, 7, 3]])

特别的，在Numpy中具有clip()函数，例子如下：
print(A)
# array([[14,13,12,11]
#        [10, 9, 8, 7]
#        [ 6, 5, 4, 3]])

print(np.clip(A,5,9))    
# array([[ 9, 9, 9, 9]
#        [ 9, 9, 8, 7]
#        [ 6, 5, 5, 5]])
这个函数的格式是clip(Array,Array_min,Array_max)，顾名思义，Array指的是将要被执行用的矩阵，而后面的最小值最大值则用于让函数判断矩阵中元素是否有比最小值小的或者比最大值大的元素，并将这些指定的元素转换为最小值或者最大值。
实际上每一个Numpy中大多数函数均具有很多变量可以操作，你可以指定行、列甚至某一范围中的元素。

Numpy 索引
一维索引 
import numpy as np
A = np.arange(3,15)

# array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
         
print(A[3])    # 6
让我们将矩阵转换为二维的，此时进行同样的操作：
A = np.arange(3,15).reshape((3,4))
"""
array([[ 3,  4,  5,  6]
       [ 7,  8,  9, 10]
       [11, 12, 13, 14]])
"""
         
print(A[2])         
# [11 12 13 14]
实际上这时的A[2]对应的就是矩阵A中第三行(从0开始算第一行)的所有元素。
二维索引 
如果你想要表示具体的单个元素，可以仿照上述的例子：
print(A[1][1])      # 8
print(A[1, 1])      # 8
利用 : 对一定范围内的元素进行切片操作：对第二行中第2到第4列元素进行切片输出（不包含第4列）
print(A[1, 1:3])    # [8 9]
print(A[2, :])    # [11 12 13 14]
print(A[:, 1])    # [4  8 12]
此时我们适当的利用for函数进行打印：
for row in A:
    print(row)
"""    
[ 3,  4,  5, 6]
[ 7,  8,  9, 10]
[11, 12, 13, 14]
"""
此时它会逐行进行打印操作。如果想进行逐列打印，就需要稍稍变化一下：
for column in A.T:
    print(column)
"""  
[ 3,  7,  11]
[ 4,  8,  12]
[ 5,  9,  13]
[ 6, 10,  14]
"""
上述表示方法即对A进行转置，再将得到的矩阵逐行输出即可得到原矩阵的逐列输出。
最后依然说一些关于迭代输出的问题：
import numpy as np
A = np.arange(3,15).reshape((3,4))
         
print(A.flatten())   
# array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

for item in A.flat:
    print(item)
    
# 3
# 4
……
# 14
这一脚本中的flatten是一个展开性质的函数，将多维的矩阵进行展开成1行的数列。而flat是一个迭代器，本身是一个object属性。

Numpy array 合并
np.vstack() 
对于一个array的合并，我们可以想到按行、按列等多种方式进行合并。首先先看一个例子：
import numpy as np
A = np.array([1,1,1])
B = np.array([2,2,2])
         
print(np.vstack((A,B)))    # vertical stack
"""
[[1,1,1]
 [2,2,2]]
"""
vertical stack本身属于一种上下合并，即对括号中的两个整体进行对应操作。此时我们对组合而成的矩阵进行属性探究：
C = np.vstack((A,B))      
print(A.shape,C.shape)

# (3,) (2,3)
np.hstack() 
利用shape函数可以让我们很容易地知道A和C的属性，从打印出的结果来看，A仅仅是一个拥有3项元素的数组（数列），而合并后得到的C是一个2行3列的矩阵。
介绍完了上下合并，我们来说说左右合并：
D = np.hstack((A,B))       # horizontal stack

print(D)
# [1,1,1,2,2,2]

print(A.shape,D.shape)
# (3,) (6,)
通过打印出的结果可以看出：D本身来源于A，B两个数列的左右合并，而且新生成的D本身也是一个含有6项元素的序列。
np.newaxis() 
说完了array的合并，我们稍稍提及一下前一节中转置操作，如果面对如同前文所述的A序列， 转置操作便很有可能无法对其进行转置（因为A并不是矩阵的属性），此时就需要我们借助其他的函数操作进行转置：
print(A[np.newaxis,:])
# [[1 1 1]]

print(A[np.newaxis,:].shape)
# (1,3)

print(A[:,np.newaxis])
"""
[[1]
[1]
[1]]
"""

print(A[:,np.newaxis].shape)
# (3,1)
此时我们便将具有3个元素的array转换为了1行3列以及3行1列的矩阵了。
结合着上面的知识，我们把它综合起来：
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
# (3,1) (3,2)
np.concatenate() 
当你的合并操作需要针对多个矩阵或序列时，借助concatenate函数可能会让你使用起来比前述的函数更加方便：
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
axis参数很好的控制了矩阵的纵向或是横向打印，相比较vstack和hstack函数显得更加方便。


Numpy array 分割
创建数据 
import numpy as np
A = np.arange(12).reshape((3, 4))
print(A)
"""
array([[ 0,  1,  2,  3],
    [ 4,  5,  6,  7],
    [ 8,  9, 10, 11]])
"""
纵向分割 
print(np.split(A, 2, axis=1))
"""
[array([[0, 1],
        [4, 5],
        [8, 9]]), 
 array([[ 2,  3],
        [ 6,  7],
        [10, 11]])]
"""
横向分割 
print(np.split(A, 3, axis=0))

# [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
错误的分割 
范例的Array只有4列，只能等量对分，因此输入以上程序代码后Python就会报错。
print(np.split(A, 3, axis=1))

# ValueError: array split does not result in an equal division
为了解决这种情况, 我们会有下面这种方式.
不等量的分割 
在机器学习时经常会需要将数据做不等量的分割，因此解决办法为np.array_split()
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

其他的分割方式 
在Numpy里还有np.vsplit()与横np.hsplit()方式可用。
print(np.vsplit(A, 3)) #等于 print(np.split(A, 3, axis=0))

# [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]


print(np.hsplit(A, 2)) #等于 print(np.split(A, 2, axis=1))
"""
[array([[0, 1],
       [4, 5],
       [8, 9]]), 
array([[ 2,  3],
        [ 6,  7],
        [10, 11]])]
"""


Numpy copy & deep copy
= 的赋值方式会带有关联性 
首先 import numpy 并建立变量, 给变量赋值。
import numpy as np

a = np.arange(4)
# array([0, 1, 2, 3])

b = a
c = a
d = b
改变a的第一个值，b、c、d的第一个值也会同时改变。
a[0] = 11
print(a)
# array([11,  1,  2,  3])
确认b、c、d是否与a相同。
b is a  # True
c is a  # True
d is a  # True
同样更改d的值，a、b、c也会改变。
d[1:3] = [22, 33]   # array([11, 22, 33,  3])
print(a)            # array([11, 22, 33,  3])
print(b)            # array([11, 22, 33,  3])
print(c)            # array([11, 22, 33,  3])
copy() 的赋值方式没有关联性 
b = a.copy()    # deep copy
print(b)        # array([11, 22, 33,  3])
a[3] = 44
print(a)        # array([11, 22, 33, 44])
print(b)        # array([11, 22, 33,  3])
此时a与b已经没有关联。














