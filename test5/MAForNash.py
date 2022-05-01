'''蜉蝣算法，write byJack旭:https://mianbaoduo.com/o/JackYM'''
'''如需其他代码请访问：链接：https://pan.baidu.com/s/1QIHWRh0bNfZRA8KCQGU8mg 提取码：1234'''

import numpy as np
import random
import math
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


''' 种群初始化函数 '''
def initial(pop, dim,size):
    X = np.zeros([pop, size,dim])
    for i in range(pop):
        X[i] =np.random.dirichlet(np.ones(dim), size=size);
    return X
def initial_V(pop, dim, ub, lb,size):
    X = np.zeros([pop,size ,dim])
    for i in range(pop):
        for k in range(size):
           for j in range(dim-1):
               X[i,k,j] = random.random() * (ub[j] - lb[j]) + lb[j]
           X[i][k][dim-1]=-sum(X[i][k])
    return X, lb, ub
def Control_step(X,V,dim):
    mina = [];
    mina.append(0)
    for j in range(dim):
        a=-X[j]/V[j];
        # print(a)
        if a>=0:
          mina.append(a);
    return mina
'''计算适应度函数'''
def CaculateFitness(X,fun):
    pop = X.shape[0]
    dim=X.shape[2]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :],dim)
    return fitness
            
'''边界检查函数'''
# def BorderCheck(X,ub,lb,pop,dim):
#     for i in range(pop):
#         for j in range(dim):
#             if X[i,j]>ub[j]:
#                 X[i,j] = ub[j]
#             elif X[i,j]<lb[j]:
#                 X[i,j] = lb[j]
#     return X
def Check(X,V,pop,dim,size):
    newX = X + V
    for i in range(pop):
        for k in range(size):
            if np.all(newX[i,k]>=0) and np.all(newX[i,k]<=1):
                if not (np.sum(newX[i,k])==1):
                    X[i, k] = (newX[i, k] - newX[i, k].min(axis=0)) / (newX[i, k].max(axis=0) - newX[i,k].min(axis=0))
                    X[i, k] = (X[i, k]) / (sum(X[i,k]));
                else:
                    X[i,k]=newX[i,k]
            else:
                mina = Control_step(X[i, k], V[i, k], dim)
                a = min(mina)
                X[i, k] = X[i, k] + a * V[i, k]
                # if not ((np.sum(X[i, k]) == 1) and np.all(X[i, k] >= 0) and np.all(X[i, k] <= 1)):
                #     X[i, k] = (X[i, k] - X[i, k].min(axis=0)) / (X[i, k].max(axis=0) - X[i, k].min(axis=0))
                #     X[i, k] = (X[i, k]) / (sum(X[i, k]));

    # print(X)
    return X
def Check2(X):
    pop = X.shape[0]
    size = X.shape[1]
    for i in range(pop):
        for k in range(size):
            if not (((np.sum(X[i, k]) == 1) and np.all(X[i, k] >= 0) and np.all(X[i, k] <= 1))):
                X[i, k] = (X[i, k] - X[i, k].min(axis=0)) / (X[i, k].max(axis=0) - X[i, k].min(axis=0))
                X[i, k] = (X[i, k]) / (sum(X[i, k]));
    return X

'''适应度排序'''
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness,index
'''根据适应度对位置进行排序'''
def SortPosition(X,index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew
'''蜉蝣算法'''
def MA(pop,dim,lb,ub,MaxIter,fun,size):
    nPop = pop #雄性数量
    nPopf = pop#雌性数量
    g = 0.52 #惯性权值
    gdamp = 1 #惯性重量阻尼比
    a1 = 0.01    #自我学习参数
    a2 = 1.5 #全局学习参数
    a3 = 0.01  #全局学习参数
    beta = 2.1  #视距系数
    dance = 3.73 #舞蹈系数
    fl = 2    #Random flight
    dace_damp = 0.8  #Damping Ratio
    fl_damp = 0.99
    nc = 30 #Number of Offsprings(also Parnets)
    nm = np.round(0.05*nPop) #Number of Mutants
    mu = 0.01                   #MuRation Rate
    VelMax = 15*(ub-lb)  #最大速度
    VelMin = -VelMax#最小速度
    
    Mayfly=initial(pop,dim,size) #雄性初始化
    fitness = CaculateFitness(Mayfly,fun) #计算适应度值
    MayflyV,VelMin,VelMax=initial_V(pop,dim,VelMax,VelMin,size) #雄性速度初始化
    index = np.argsort(fitness, axis=0)
    fitnessBest = fitness[index[0]]
    MayflyBest = copy.copy(Mayfly[index[0],:])

    Mayflyf=initial(pop,dim,size) #雌性初始化
    fitnessf = CaculateFitness(Mayflyf,fun) #计算适应度值
    MayflyfV,VelMin,VelMax=initial_V(pop,dim,VelMax,VelMin,size) #雌性速度初始化
    index = np.argsort(fitnessf, axis=0)
    fitnessfBest = fitnessf[index[0]]
    MayflyfBest = copy.copy(Mayflyf[index[0],:])
    
    #记录最优值
    GbestScore = np.inf
    GbestPositon = np.zeros([2,dim])
    for i in range(pop):
        if fitness[i]<GbestScore:
            GbestScore = copy.copy(fitness[i])
            GbestPositon= copy.copy(Mayfly[i,:])
        if fitnessf[i]<GbestScore:
            GbestScore = copy.copy(fitnessf[i])
            GbestPositon = copy.copy(Mayflyf[i,:])
    print(GbestPositon)
    Curve = np.zeros([MaxIter,1])

    for t in range(MaxIter):
        print("第"+str(t)+"次迭代")
        #更新雌性
        for i in range(nPopf):
            e = np.random.uniform(-1,1,dim)
            rmf = np.linalg.norm(Mayfly[i,:]-Mayflyf[i,:])
            if fitnessf[i]>fitness[i]:
                MayflyfV[i,:]=g*MayflyfV[i,:]+a3*np.exp(-beta*rmf**2)*(Mayfly[i,:]-Mayflyf[i,:])
            else:
                MayflyfV[i,:] = g*MayflyfV[i,:]+fl*e

        # Mayflyf=Check(Mayflyf,MayflyfV,pop,dim,size)
        fitnessf = CaculateFitness(Mayflyf,fun) #计算适应度值
        #更新雄性
        for i in range(nPop):
            rpbest = np.linalg.norm(MayflyBest-Mayflyf[i,:])
            rgbest = np.linalg.norm(GbestPositon - Mayflyf[i,:])
            e = np.random.uniform(-1,1,dim)
            if fitness[i]>fitnessf[i]:
                MayflyV[i,:] = g*MayflyV[i,:]+a1*np.exp(-beta*rpbest**2)*(MayflyBest-Mayfly[i,:])\
                +a2*np.exp(-beta*rgbest**2)*(GbestPositon-Mayfly[i,:])
            else:
                MayflyV[i,:] = g*MayflyV[i,:]+dance*e
            

        Mayfly = Check(Mayfly, MayflyV, pop, dim, size)
        fitness = CaculateFitness(Mayfly,fun) #计算适应度值   
        for i in range(pop):
            if fitness[i]<GbestScore:
                GbestScore = copy.copy(fitness[i])
                GbestPositon= copy.copy(Mayfly[i,:])
            if fitnessf[i]<GbestScore:
                GbestScore = copy.copy(fitnessf[i])
                GbestPositon= copy.copy(Mayflyf[i,:])
        # print(GbestPositon)
        fitness,sortIndex = SortFitness(fitness) #对适应度值排序
        Mayfly = SortPosition(Mayfly,sortIndex) #种群排序   
       
        fitnessf,sortIndex = SortFitness(fitnessf) #对适应度值排序
        Mayflyf = SortPosition(Mayflyf,sortIndex) #种群排序  

         
       #交配
        MayflyOffspring = np.zeros([nc,size,dim])
        for k in range(0,nc,2):
           L = np.random.uniform(-1,1,dim)
           MayflyOffspring[k,:] = L*Mayfly[k,:] + (1-L)*Mayflyf[k,:]
           L = np.random.uniform(-1,1,dim)
           MayflyOffspring[k+1,:] = L*Mayflyf[k,:] + (1-L)*Mayfly[k,:]

        # MayflyOffspring =BorderCheck(MayflyOffspring,ub,lb,nc,dim)
        MayflyOffspring = Check2(MayflyOffspring)
        # print(MayflyOffspring)
        fitnessOffspring = CaculateFitness(MayflyOffspring,fun) #计算适应度值  
        #更新最优值
        for i in range(nc):
            if fitnessOffspring[i]<GbestScore:
                GbestScore = copy.copy(fitnessOffspring[i])
                GbestPositon= copy.copy(MayflyOffspring[i,:])
        # print(GbestPositon)
        #合并种群，更新雌性和雄性
        NewMayfly= np.vstack((Mayfly,MayflyOffspring))
        # NewMayfly=Check2(NewMayfly)
        # print(NewMayfly)
        Newfitness = CaculateFitness(NewMayfly,fun) #计算适应度值
        NewMayflyf = np.vstack((Mayflyf,MayflyOffspring))
        # NewMayflyf = Check2(NewMayflyf)
        # print(NewMayflyf)
        Newfitnessf = CaculateFitness(NewMayflyf,fun) #计算适应度值
        Newfitness,sortIndex = SortFitness(Newfitness) #对适应度值排序
        NewMayfly = SortPosition(NewMayfly,sortIndex) #种群排序   
        Newfitnessf,sortIndex = SortFitness(Newfitnessf) #对适应度值排序
        NewMayflyf = SortPosition(NewMayflyf,sortIndex) #种群排序  
        Mayfly = copy.copy(NewMayfly[0:nPop,:])
        fitness = copy.copy(Newfitness[0:nPop,:])
        Mayflyf = copy.copy(NewMayflyf[0:nPop,:])
        fitnessf = copy.copy(Newfitnessf[0:nPop,:])
         
         
         
            
        Curve[t] = GbestScore

    
    return GbestScore,GbestPositon,Curve









