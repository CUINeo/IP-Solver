# integer programming implemented by branch and bound

import math
import time
import numpy as np
from scipy.optimize import linprog
from linprog_new import simplex
from copy import deepcopy
from case_process import process
from case_process import get_case


def check_int(x, epsilon=1E-6):
    return x-math.floor(x) < epsilon or math.ceil(x)-x < epsilon


def round_int(x, epsilon=1E-6):
    return math.floor(-x + epsilon)


def build_bounds(branch_node, bounds):
    bounds_list = [list(bounds[i]) for i in range(len(bounds))]
    br_num = len(branch_node[0][0])
    
    # add floor/ceil bound in branch to bounds
    for i in range(br_num):
        x_idx = branch_node[0][0][i]
        bound_type = branch_node[0][1][i][0]
        bound_val = branch_node[0][1][i][1]
        
        if bound_type == 1:  # floor bound
            # if bounds_list[x_idx][0] is None or bound_val > bounds_list[x_idx][0]:
            bounds_list[x_idx][0] = bound_val
        elif bound_type == -1:  # ceil bound
            # if bounds_list[x_idx][1] is None or bound_val < bounds_list[x_idx][0]:
            bounds_list[x_idx][1] = bound_val
    
    # convert to tuple
    branch_bounds = [tuple(bounds_list[i]) for i in range(len(bounds))]
    return tuple(branch_bounds)


def intprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):  
    # set the upper bound and lower bound
    global x_relaxed
    fun_ub = float('inf')
    fun_lb = float('inf')
    
    N = len(c)
    branches = [0]
    
    x_opt = []
    fun = []
    
    if bounds is None:  # without the parameter@bounds
        bounds = tuple([(None, None) for i in range(N)])
        
    # the result in this branch has not been derived
    while len(branches) != 0:
        integer_vec = [1 for i in range(N)]  # x_i is integer(precision decided by epsilon in check_int)
        # print("iter 1")
        # Solve relaxtion problem
        if branches[0] == 0:
            # branches[0] exactly represents the original problem
            c_, A_ub_, b_ub_, A_eq_, b_eq_ = deepcopy(c), deepcopy(A_ub), deepcopy(b_ub), deepcopy(A_eq), deepcopy(b_eq)
            bounds_ = bounds
            res_fun, res_x, res_status = simplex(c_, A_ub_, b_ub_, A_eq_, b_eq_, bounds_)
            # c, A_ub, b_ub, A_eq, b_eq = c[:N], A_ub[:N], b_ub[:N], A_eq[:N], b_eq[:N]

        else:
            # build bounds from branch
            branch_bounds = build_bounds(branches, bounds)
            # print(len(A_eq[0]))
            c_, A_ub_, b_ub_, A_eq_, b_eq_ = deepcopy(c), deepcopy(A_ub), deepcopy(b_ub), deepcopy(A_eq), deepcopy(b_eq)
            branch_bounds_ = branch_bounds
            res_fun, res_x, res_status = simplex(c_, A_ub_, b_ub_, A_eq_, b_eq_, branch_bounds_)
            # c, A_ub, b_ub, A_eq, b_eq = c[:N], A_ub[:N], b_ub[:N], A_eq[:N], b_eq[:N]
                
        if res_status == 0:
            # update the lower bound (Min Opt assumed)
            x_relaxed = res_x
            fun_lb = res_fun
        else:
            fun_lb = float('inf')

        # branch and bound
        if fun_lb < fun_ub:
            # update int_vec to check x_relaxed
            for i in range(N):
                if not check_int(x_relaxed[i]):
                    integer_vec[i] = 0

            # debug of the noninteger variable
            # for i in range(len(integer_vec)):
            #     if integer_vec[i] != 1:
            #         print(i)

            if sum(integer_vec) != N:
                # trick: set the optimized variable as the root
                # find the first non-integer variable in c-vector
                x_idx = -1
                for i in range(N):
                    if(c[i] != 0 and integer_vec[i] == 0):
                        x_idx = i
                if(x_idx == -1):
                    x_idx = integer_vec.index(0)
                if branches[0] == 0:
                    branches.append([[x_idx], [[-1, math.floor(x_relaxed[x_idx])]]])
                    branches.append([[x_idx], [[1, math.ceil(x_relaxed[x_idx])]]])
                else:
                    branches[0][0].append(x_idx)
                    branches.append(deepcopy(branches[0]))
                    branches.append(deepcopy(branches[0]))
                    branches[-1][1].append([-1, math.floor(x_relaxed[x_idx])])
                    branches[-2][1].append([1, math.ceil(x_relaxed[x_idx])])
            else:
                fun_ub = fun_lb
                x_opt = x_relaxed

        # branch-cutting
        branches.pop(0)
        if len(branches) == 0:
            if len(x_opt) == 0:
                pass
            else:
                fun = np.dot(np.array(c).reshape([1, N]), x_opt.reshape([N, 1]))[0][0]
                
    return x_opt, round_int(fun)


if __name__ == "__main__":

    start = time.perf_counter()

    N, c, A_ub, b_ub, A_eq, b_eq, bounds = process(get_case("TestCase5.txt"))
    print(intprog(c, A_ub, b_ub, A_eq, b_eq, bounds))

    # c = [-600, -400]
    # A_ub = [[6, 8], [10, 5], [11, 8]]
    # b_ub = [120, 100, 130]
    # A_eq = [[0, 0]]
    # b_eq = [0]
    # bounds = ([0, float('inf')], [0, float('inf')])
    # print(intprog(c, A_ub, b_ub, A_eq, b_eq, bounds))

    elapsed = (time.perf_counter() - start)
    print('Time used: ', elapsed)
