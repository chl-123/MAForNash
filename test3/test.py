from pulp import *

# 1. 建立问题
prob = LpProblem("Problem", LpMinimize)

# 2. 建立变量
x1 = LpVariable("x1",0)
x2 = LpVariable("x2",0)
x3 = LpVariable("x3",0)
x4 = LpVariable("x4",0)
x5 = LpVariable("x5",0)

# 3. 设置目标函数 z
prob += (x1 + x2+ x3+ x4+ x5)

# 4. 施加约束
# prob += x1 + x2+ x3+ x4+ x5==1
prob += 7.99*x1 + 7.97*x2+ 7.90*x3+ 7.70*x4+ 7.23*x5 >=1
prob += 7.49*x1 + 7.80*x2+ 8.00*x3+ 8.02*x4+ 7.69*x5 >=1
prob += 6.99*x1 + 7.64*x2+ 8.11*x3+ 8.32*x4+ 8.14*x5 >=1
prob += 6.49*x1 + 7.47*x2+ 8.22*x3+ 8.62*x4+ 8.59*x5 >=1
prob += 5.99*x1 + 7.31*x2+ 8.32*x3+ 8.93*x4+ 9.04*x5 >=1

# 5. 求解
prob.solve()

# 6. 打印求解状态
print("Status:", LpStatus[prob.status])

# 8. 打印最优解的目标函数值
print("z= ", value(prob.objective))

# 7. 打印出每个变量的最优值
for v in prob.variables():
    print(v.name, "=", v.varValue)

