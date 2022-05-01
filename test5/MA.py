'''蜉蝣算法，write byJack旭:https://mianbaoduo.com/o/JackYM'''
'''如需其他代码请访问：链接：https://pan.baidu.com/s/1QIHWRh0bNfZRA8KCQGU8mg 提取码：1234'''

import numpy as np
import random
import math
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


''' 种群初始化函数 '''
def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random()*(ub[j] - lb[j]) + lb[j]
    
    return X,lb,ub
            
'''边界检查函数'''
def BorderCheck(X,ub,lb,pop,dim):
    for i in range(pop):
        for j in range(dim):
            if X[i,j]>ub[j]:
                X[i,j] = ub[j]
            elif X[i,j]<lb[j]:
                X[i,j] = lb[j]
    return X
    
    
'''计算适应度函数'''
def CaculateFitness(X,fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness

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
def MA(pop,dim,lb,ub,MaxIter,fun):
    nPop = pop #雄性数量
    nPopf = pop#雌性数量
    g = 0.8 #惯性权值
    gdamp = 1 #惯性重量阻尼比
    a1 = 1    #自我学习参数
    a2 = 1.5  #全局学习参数
    a3 = 1.5  #全局学习参数
    beta = 2  #视距系数
    dance = 5 #舞蹈系数
    fl = 1    #Random flight
    dace_damp = 0.8  #Damping Ratio
    fl_damp = 0.99
    nc = 20 #Number of Offsprings(also Parnets)
    nm = np.round(0.05*nPop) #Number of Mutants
    mu = 0.01                   #MuRation Rate
    VelMax = 0.1*(ub-lb)  #最大速度
    VelMin = -VelMax#最小速度
    
    Mayfly,lb,ub=initial(pop,dim,ub,lb) #雄性初始化
    fitness = CaculateFitness(Mayfly,fun) #计算适应度值
    MayflyV,VelMin,VelMax=initial(pop,dim,VelMax,VelMin) #雄性速度初始化
    index = np.argsort(fitness, axis=0)
    fitnessBest = fitness[index[0]]
    MayflyBest = copy.copy(Mayfly[index[0],:])
    
    
    Mayflyf,lb,ub=initial(pop,dim,ub,lb) #雌性初始化
    fitnessf = CaculateFitness(Mayflyf,fun) #计算适应度值
    MayflyfV,VelMin,VelMax=initial(pop,dim,VelMax,VelMin) #雌性速度初始化
    index = np.argsort(fitnessf, axis=0)
    fitnessfBest = fitnessf[index[0]]
    MayflyfBest = copy.copy(Mayflyf[index[0],:])
    
    #记录最优值
    GbestScore = np.inf
    GbestPositon = np.zeros([1,dim])
    for i in range(pop):
        if fitness[i]<GbestScore:
            GbestScore = copy.copy(fitness[i])
            GbestPositon[0,:] = copy.copy(Mayfly[i,:])
        if fitnessf[i]<GbestScore:
            GbestScore = copy.copy(fitnessf[i])
            GbestPositon[0,:] = copy.copy(Mayflyf[i,:])
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
                
        MayflyfV=BorderCheck(MayflyfV,VelMax,VelMin,pop,dim) 
        Mayflyf = Mayflyf + MayflyfV
        Mayflyf =BorderCheck(Mayflyf,ub,lb,pop,dim)
        fitnessf = CaculateFitness(Mayflyf,fun) #计算适应度值
        #更新雄性
        for i in range(nPop):
            rpbest = np.linalg.norm(MayflyBest-Mayflyf[i,:])
            rgbest = np.linalg.norm(GbestPositon[0,:] - Mayflyf[i,:])
            e = np.random.uniform(-1,1,dim)
            if fitness[i]>fitnessf[i]:
                MayflyV[i,:] = g*MayflyV[i,:]+a1*np.exp(-beta*rpbest**2)*(MayflyBest-Mayfly[i,:])\
                +a2*np.exp(-beta*rgbest**2)*(GbestPositon[0,:]-Mayfly[i,:])
            else:
                MayflyV[i,:] = g*MayflyV[i,:]+dance*e
            
        MayflyV=BorderCheck(MayflyV,VelMax,VelMin,pop,dim) 
        Mayfly = Mayfly + MayflyV
        Mayfly =BorderCheck(Mayfly,ub,lb,pop,dim)
        fitness = CaculateFitness(Mayfly,fun) #计算适应度值   
        for i in range(pop):
            if fitness[i]<GbestScore:
                GbestScore = copy.copy(fitness[i])
                GbestPositon[0,:] = copy.copy(Mayfly[i,:])
            if fitnessf[i]<GbestScore:
                GbestScore = copy.copy(fitnessf[i])
                GbestPositon[0,:] = copy.copy(Mayflyf[i,:])
        fitness,sortIndex = SortFitness(fitness) #对适应度值排序
        Mayfly = SortPosition(Mayfly,sortIndex) #种群排序   
       
        fitnessf,sortIndex = SortFitness(fitnessf) #对适应度值排序
        Mayflyf = SortPosition(Mayflyf,sortIndex) #种群排序  

         
       #交配
        MayflyOffspring = np.zeros([nc,dim])
        for k in range(0,nc,2):
           L = np.random.uniform(-1,1,dim)
           MayflyOffspring[k,:] = L*Mayfly[k,:] + (1-L)*Mayflyf[k,:]
           L = np.random.uniform(-1,1,dim)
           MayflyOffspring[k+1,:] = L*Mayflyf[k,:] + (1-L)*Mayfly[k,:]
        
        MayflyOffspring =BorderCheck(MayflyOffspring,ub,lb,nc,dim)
        fitnessOffspring = CaculateFitness(MayflyOffspring,fun) #计算适应度值  
        #更新最优值
        for i in range(nc):
            if fitnessOffspring[i]<GbestScore:
                GbestScore = copy.copy(fitnessOffspring[i])
                GbestPositon[0,:] = copy.copy(MayflyOffspring[i,:])
        
        #合并种群，更新雌性和雄性
        NewMayfly= np.vstack((Mayfly,MayflyOffspring))
        Newfitness = CaculateFitness(NewMayfly,fun) #计算适应度值    
        NewMayflyf = np.vstack((Mayflyf,MayflyOffspring))
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









