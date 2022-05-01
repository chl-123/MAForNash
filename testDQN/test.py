'''蜉蝣算法，write byJack旭:https://mianbaoduo.com/o/JackYM'''


'''如需其他代码请访问：链接：https://pan.baidu.com/s/1QIHWRh0bNfZRA8KCQGU8mg 提取码：1234'''
import numpy as np
from matplotlib import pyplot as plt
import MAForNash

'''定义目标函数用户'''
def fun7(X,dim):#定义适应度函数
    # A=np.array(
    #     [
    #         [3, 1 ,6],
    #         [0, 0, 4],
    #         [1, 2, 5]
    #     ]
    # )
    # B = np.array(
    #     [
    #         [3, 0, 1],
    #         [1, 0, 2],
    #         [6, 4, 5]
    #     ]
    # )

    A=np.array(
        [
            [1, 0 ,0],
            [0, 1, 0],
            [0, 0, 1]
        ]
    )
    B = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ]
    )

    # B=np.transpose(A)

    y=X[1:]
    print(y)
    x=X[0:1]
    print(x)
    y=np.transpose(y);
    V1=x.dot(A).dot(y);
    V2 = x.dot(B).dot(y);
    print(V2)
    print(V1)
    a=[];
    a.append(0)
    b=[]
    b.append(0)
    for i in range(0,dim):
        print(A[i:i+1,])
        k=A[i:i+1,].dot(y)[0][0]-V1[0][0];
        # print(k)
        a.append(k)
    for i in range(0,dim):
        V=x.dot(B[:,i:i+1])[0][0]-V2[0][0];
        print(B[:,i:i+1])
        # print(V)
        b.append(V)
    print(a)
    print(b)
    return max(a)+max(b)
X= A=np.array(
        [
            [0.333, 0.33 ,0.333],
            [0.33, 0.33, 0.333]
        ]
    )
print(fun7(X,3))