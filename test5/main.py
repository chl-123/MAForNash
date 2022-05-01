import numpy as np
from matplotlib import pyplot as plt
import Tent_SSA

'''定义目标函数用户可选fun1 - fun6 , 也可以自己定义自己的目标函数'''
def fun1(X):
        O=np.sum(X*X)
        return O

def fun2(X):
    O=np.sum(np.abs(X))+np.prod(np.abs(X))
    return O

def fun3(X):
    O=0
    for i in range(len(X)):
        O=O+np.square(np.sum(X[0:i+1]))   
    return O

def fun4(X):
    O=np.max(np.abs(X))
    return O

def fun5(X):
    X_len = len(X)
    O = np.sum(100 * np.square(X[1:X_len] - np.square(X[0:X_len - 1]))) + np.sum(np.square(X[0:X_len - 1] - 1))
    return O

def fun6(X):
    O=np.sum(np.square(np.abs(X+0.5)))
    return O


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

    # A=np.array(
    #     [
    #         [1, 0 ,0],
    #         [0, 1, 0],
    #         [0, 0, 1]
    #     ]
    # )
    # B = np.array(
    #     [
    #         [0, 1, 0],
    #         [0, 0, 1],
    #         [1, 0, 0]
    #     ]
    # )
    A = np.array(
        [
            [1, 2, 0],
            [0, 1,2],
            [2, 0, 1]
        ]
    )
    B = np.array(
        [
            [1, 0, 2],
            [2, 1, 0],
            [0, 2, 1]
        ]
    )
    # B = np.array(
    #     [
    #         [0, 4, 5],
    #         [4, 0, 5],
    #         [3, 3, 6]
    #     ]
    # )
    # A = np.array(
    #     [
    #         [4, 0, 3],
    #         [0, 4,3],
    #         [5, 5, 6]
    #     ]
    # )


    # B=np.transpose(A)

    y=X[1:]
    x=X[0:1]
    y=np.transpose(y);
    V1=x.dot(A).dot(y);
    V2 = x.dot(B).dot(y);
    # print(V2)
    # print(V1)
    a=[];
    a.append(0)
    b=[]
    b.append(0)
    for i in range(0,dim):
        k=A[i:i+1,].dot(y)[0][0]-V1[0][0];
        # print(k)
        a.append(k)
    for i in range(0,dim):
        V=x.dot(B[:,i:i+1])[0][0]-V2[0][0];
        # print(V)
        b.append(V)
    return max(a)+max(b)


'''主函数 '''
#设置参数
pop = 30 #种群数量
MaxIter = 1000 #最大迭代次数
size=2
dim = 3 #维度
lb = -100*np.ones([dim, 1]) #下边界
ub = 100*np.ones([dim, 1])#上边界
#适应度函数选择
fobj = fun7
GbestScore,GbestPositon,Curve = Tent_SSA.Tent_SSA(pop,dim,size,lb,ub,MaxIter,fobj)
print('最优适应度值：',GbestScore)
print('最优解：',GbestPositon)

#绘制适应度曲线
plt.figure(1)
plt.semilogy(Curve,'r-',linewidth=2)
plt.xlabel('Iteration',fontsize='medium')
plt.ylabel("Fitness",fontsize='medium')
plt.grid()
plt.title('Tent_SSA',fontsize='large')
plt.show()