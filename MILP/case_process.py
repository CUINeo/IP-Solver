"""
get_case: get testcases from file
process_case: process testcase, return N, c, A_ub, b_ub, A_eq, b_eq, Bounds
"""
import re
import numpy as np

file = 0


def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False


def get_case(path):
    global file

    if path.__contains__('3'):
        file = 3
    elif path.__contains__('4'):
        file = 4
    elif path.__contains__('5'):
        file = 5

    with open(path, 'r') as f:
        return f.readlines()


def process(lines):
    # define modes
    obj_func_mode = 1
    constraint_mode = 2
    variable_bound_mode = 3
    integer_def_mode = 4

    # initialize mode
    mode = obj_func_mode

    # initialize return value
    if file == 3 or file == 4:
        N = 442
    elif file == 5:
        N = 1111

    c = []
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    bounds = [(0, float('inf')) for i in range(N)]

    for line in lines:
        line = line.replace('\n', '')
        if line.endswith(';'):
            line = line.replace(';', '')

        # empty line
        if line == '':
            continue

        # switch modes
        if line == "/* Objective function */":
            mode = obj_func_mode
            continue
        elif line == "/* Constraints */":
            mode = constraint_mode
            continue
        elif line == "/* Variable bounds */":
            mode = variable_bound_mode
            continue
        elif line == "/* Integer definitions */":
            mode = integer_def_mode
            continue

        # branch according to mode
        if mode == obj_func_mode:
            # objective function
            parts = line.split(" ")

            i = 0
            for obj in parts:
                if obj.startswith("+"):
                    i = int(re.findall(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$', obj)[0])
                    c = [0]*N
                    c[i-1] = 1
                if obj.startswith("-"):
                    i = int(re.findall(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$', obj)[0])
                    c = [0]*N
                    c[i-1] = -1

            if line.startswith('max:'):
                    c[i-1] *= -1

        elif mode == constraint_mode:
            # constraints
            A = [0]*N
            b = 0
            flag = 0
            parts = line.split(' ')

            for s in parts:
                if (s.startswith('-') or s.startswith('+')) and s.__contains__('C'):
                    # var
                    i = int(re.findall(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$', s)[0])
                    if s.startswith('-'):
                        A[i-1] = -1
                    elif s.startswith('+'):
                        A[i-1] = 1
                elif s == '=':
                    # eq
                    flag = 0
                elif s == '<=':
                    # ub
                    flag = 1
                elif s == '>=':
                    # lb
                    flag = 2
                elif is_number(s):
                    # b
                    b = int(s)

            if flag == 0:
                # eq
                A_eq.append(A)
                b_eq.append(b)
            elif flag == 1:
                # ub
                A_ub.append(A)
                b_ub.append(b)
            elif flag == 2:
                # lb
                A = [-i for i in A]
                A_ub.append(A)
                b_ub.append(-b)

        elif mode == variable_bound_mode:
            # variable bounds
            i = 1

            parts = line.split(' ')

            for s in parts:
                if is_number(s):
                    b = int(s)
                    bounds[i-1] = (b, b)
                elif s.startswith('C'):
                    i = int(re.findall(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$', s)[0])

        elif mode == integer_def_mode:
            # integer definition
            continue

    if file == 4:
        b_eq.append(2)
        A = [0]*442
        A[0] = 5
        A[1] = 3
        A[2] = 1
        A[3] = 1
        A[4] = 1
        A[5] = 1
        A_eq.append(A)

    return N, c, A_ub, b_ub, A_eq, b_eq, bounds


if __name__ == "__main__":
    N, c, A_ub, b_ub, A_eq, b_eq, bounds = process(get_case("TestCase5.txt"))

    np.savetxt('matrix\\5_C.txt', np.array(c))
    np.savetxt('matrix\\5_Aub.txt', np.array(A_ub))
    np.savetxt('matrix\\5_bub.txt', np.array(b_ub))
    np.savetxt('matrix\\5_Aeq.txt', np.array(A_eq))
    np.savetxt('matrix\\5_beq.txt', np.array(b_eq))
    np.savetxt('matrix\\5_bounds.txt', np.array(bounds))

    # print(np.loadtxt('TC1_C.txt'))
