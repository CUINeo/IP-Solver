# smallcase1
c = [-3, -2]
A_ub = [[2, 3], [1, 0.5]]
b_ub = [14, 4.5]
A_eq = [[0, 0]]
b_eq = [0]
bounds = ([0, float('inf')], [0, float('inf')])

# smallcase2
c = [-3, 2, -5]
A_ub = [[1, 2, -1], [1, 4, 1], [1, 1, 0], [0, 4, 1]]
b_ub = [2, 4, 3, 6]
A_eq = [[0, 0, 0]]
b_eq = [0]
bounds = ([0, 1], [0, 1], [0, 1])

# smallcase3
c = [-800, -300]
A_ub = [[6, 8], [10, 5]]
b_ub = [120, 100]
A_eq = [[0, 0]]
b_eq = [0]
bounds = ([0, float('inf')], [0, float('inf')])

# smallcase4
c = [-600, -400]
A_ub = [[6, 8], [10, 5], [11, 8]]
b_ub = [120, 100, 130]
A_eq = [[0, 0]]
b_eq = [0]
bounds = ([0, float('inf')], [0, float('inf')])
