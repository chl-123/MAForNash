import numpy as np
import random
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


''' Tent种群初始化函数 '''
# def initial(pop, dim, ub, lb):
#     X = np.zeros([pop, dim])
#     a = 0.7
#     x0=random.random()#初始点
#     print(x0)
#     for i in range(pop):
#         for j in range(dim):
#             if x0<a:
#                 TentValue = x0/a
#             if x0>=a:
#                 TentValue = (1-x0)/(1-a)
#             X[i, j] = TentValue*(ub[j] - lb[j]) + lb[j]
#             if X[i,j]>ub[j]:
#                 X[i, j] = ub[j]
#             if X[i,j]<lb[j]:
#                 X[i, j] = lb[j]
#             x0 = TentValue
#     print(X)
#     return X,lb,ub
def initial(pop, dim,size):
    X = np.zeros([pop, size,dim])
    for i in range(pop):
        X[i] =np.random.dirichlet(np.ones(dim), size=size);
    return X
'''边界检查函数'''
# def BorderCheck(X,ub,lb,pop,dim):
#     for i in range(pop):
#         for j in range(dim):
#             if X[i,j]>ub[j]:
#                 X[i,j] = ub[j]
#             elif X[i,j]<lb[j]:
#                 X[i,j] = lb[j]
#     return X
def BorderCheck(X):
    pop = X.shape[0]
    size = X.shape[1]
    for i in range(pop):
        for k in range(size):
            if not (((np.sum(X[i, k]) == 1) and np.all(X[i, k] >= 0) and np.all(X[i, k] <= 1))):
                X[i, k] = (X[i, k] - X[i, k].min(axis=0)) / (X[i, k].max(axis=0) - X[i, k].min(axis=0))
                X[i, k] = (X[i, k]) / (sum(X[i, k]));
    return X
    
'''计算适应度函数'''
# def CaculateFitness(X,fun):
#     pop = X.shape[0]
#     fitness = np.zeros([pop, 1])
#     for i in range(pop):
#         fitness[i] = fun(X[i, :])
#     return fitness
def CaculateFitness(X,fun):
    pop = X.shape[0]
    dim=X.shape[2]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :],dim)
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

'''麻雀发现者更新'''
def PDUpdate(X,PDNumber,ST,Max_iter,dim):
    X_new  = copy.copy(X)
    R2 = random.random()
    for j in range(PDNumber):
        if R2<ST:
            X_new[j,:] = X[j,:]*np.exp(-j/(random.random()*Max_iter))
        else:
            X_new[j,:] = X[j,:] + np.random.randn()*np.ones([1,dim])
    return X_new
        
'''麻雀加入者更新'''            
def JDUpdate(X,PDNumber,pop,dim):
    X_new = copy.copy(X)
    for j in range(PDNumber+1,pop):
         if j>(pop - PDNumber)/2 + PDNumber:
             X_new[j,:]= np.random.randn()*np.exp((X[-1,:] - X[j,:])/j**2)
         else:
             #产生-1，1的随机数
             A = np.ones([dim,1])
             for a in range(dim):
                 if(random.random()>0.5):
                     A[a]=-1       
         AA = np.dot(A,np.linalg.inv(np.dot(A.T,A)))
         X_new[j,:]= X[1,:] + np.abs(X[j,:] - X[1,:])*AA.T
           
    return X_new                    
            
'''危险更新'''   
def SDUpdate(X,pop,SDNumber,fitness,BestF):
    X_new = copy.copy(X)
    Temp = range(pop)
    RandIndex = random.sample(Temp, pop)
    SDchooseIndex = RandIndex[0:SDNumber]
    for j in range(SDNumber):
        if fitness[SDchooseIndex[j]]>BestF:
            X_new[SDchooseIndex[j],:] = X[0,:] + np.random.randn()*np.abs(X[SDchooseIndex[j],:] - X[1,:])
        elif fitness[SDchooseIndex[j]] == BestF:
            K = 2*random.random() - 1
            X_new[SDchooseIndex[j],:] = X[SDchooseIndex[j],:] + K*(np.abs( X[SDchooseIndex[j],:] - X[-1,:])/(fitness[SDchooseIndex[j]] - fitness[-1] + 10E-8))
    return X_new
              
    

'''麻雀搜索算法'''
def Tent_SSA(pop,dim,size,lb,ub,Max_iter,fun):
    ST = 80 #预警值
    PD = 1 #发现者的比列，剩下的是加入者
    SD = 1 #意识到有危险麻雀的比重
    PDNumber = int(pop*PD) #发现者数量
    SDNumber = int(pop*SD) #意识到有危险麻雀数量
    X = initial(pop, dim,size) #初始化种群
    fitness = CaculateFitness(X,fun) #计算适应度值
    fitness,sortIndex = SortFitness(fitness) #对适应度值排序
    X = SortPosition(X,sortIndex) #种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([2,dim])
    GbestPositon= copy.copy(X[0,:])
    # for i in range(pop):
    #     if fitness[i]<GbestScore:
    #         GbestScore = copy.copy(fitness[i])
    #         GbestPositon= copy.copy(Mayfly[i,:])
    #     if fitnessf[i]<GbestScore:
    #         GbestScore = copy.copy(fitnessf[i])
    #         GbestPositon = copy.copy(Mayflyf[i,:])
    Curve = np.zeros([Max_iter,1])
    for i in range(Max_iter):
        
        BestF = fitness[0]
        print(BestF)
        X = PDUpdate(X,PDNumber,ST,Max_iter,dim)#发现者更新
        
        X = JDUpdate(X,PDNumber,pop,dim) #加入者更新
        
        X = SDUpdate(X,pop,SDNumber,fitness,BestF) #危险更新
        # print(X)
        X = BorderCheck(X) #边界检测

        fitness = CaculateFitness(X,fun) #计算适应度值
        fitness,sortIndex = SortFitness(fitness) #对适应度值排序
        X = SortPosition(X,sortIndex) #种群排序
        if(fitness[0]<=GbestScore): #更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon= copy.copy(X[0,:])
        Curve[i] = GbestScore
    
    return GbestScore,GbestPositon,Curve








