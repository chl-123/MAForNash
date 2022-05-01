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
    # A = np.array(
    #     [
    #         [1, 2, 0],
    #         [0, 1,2],
    #         [2, 0, 1]
    #     ]
    # )
    # B = np.array(
    #     [
    #         [1, 0, 2],
    #         [2, 1, 0],
    #         [0, 2, 1]
    #     ]
    # )
    B = np.array(
        [
            [0, 4, 5],
            [4, 0, 5],
            [3, 3, 6]
        ]
    )
    A = np.array(
        [
            [4, 0, 3],
            [0, 4,3],
            [5, 5, 6]
        ]
    )


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
size=2
MaxIter = 3000 #最大迭代次数
dim = 3 #维度
lb = -10*np.ones([dim, 1]) #下边界
ub = 10*np.ones([dim, 1])#上边界
#适应度函数选择
fobj = fun7
GbestScore,GbestPositon,Curve = MAForNash.MA(pop, dim,lb, ub, MaxIter, fobj,size)
print('最优适应度值：',GbestScore)
print('最优解：',GbestPositon)

#绘制适应度曲线
plt.figure(1)
plt.semilogy(Curve,'r-',linewidth=1.45)
plt.xlabel('Iteration',fontsize='medium')
plt.ylabel("Fitness",fontsize='medium')
plt.grid()
plt.title('MA',fontsize='large')
plt.show()
