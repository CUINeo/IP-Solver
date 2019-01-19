from case_process import *
import numpy as np
import time

display = True

tol = 10e-6

"""
Maximize:

    c @ x

Subject to::

    A_ub @ x <= b_ub
    A_eq @ x == b_eq
     lb <= x <= ub

where ``lb = 0`` and ``ub = None`` unless specified.

Parameters
----------
c : 1D array
    Coefficients of the linear objective function to be minimized.
A_ub : 2D array, optional
    2D array such that ``A_ub @ x`` gives the values of the upper-bound
    inequality constraints at ``x``.
b_ub : 1D array, optional
    1D array of values representing the upper-bound of each inequality
    constraint (row) in ``A_ub``.
A_eq : 2D, optional
    2D array such that ``A_eq @ x`` gives the values of the equality
    constraints at ``x``.
b_eq : 1D array, optional
    1D array of values representing the RHS of each equality constraint
    (row) in ``A_eq``.
lb/ub : optional
    ``(min, max)`` pairs for each element in ``x``, defining
    the bounds on that parameter. Use None for one of ``min`` or
    ``max`` when there is no bound in that direction. By default
    bounds are ``(0, None)`` (non-negative).
    If a sequence containing a single tuple is provided, then ``min`` and
    ``max`` will be applied to all variables in the problem.
method : str, optional
    currently only supports 'simplex'.
"""

INF = float("inf")

SUCCESS = 0
INFEASIBLE = 1


def not_zero(x):
    return np.abs(x) > tol


def retrieve_x(n, base, baseRow, A_eq):
    # Given A_eq and base xis, find base solution

    X = np.zeros(n)
    X[(base == 1)] = A_eq[baseRow[(base == 1)], n]
    return X


def __simplex(c, A_eq, b_eq, base, base_row):
    m, n = A_eq.shape
    A_eq = np.column_stack((A_eq, b_eq))
    c = np.concatenate((c, np.zeros(1)))

    c -= c[np.hstack([(base == 1), False])].dot(A_eq[base_row[(base == 1)], :])

    while True:
        # check if finish
        finished, status = finish(base, c, A_eq)
        if finished and status == SUCCESS:
            # retrieve the optimal xis
            X = retrieve_x(n, base, base_row, A_eq)
            return -c[n], X, base, base_row, A_eq, SUCCESS
        elif finished and status == INFEASIBLE:
            # solution is unbounded
            return None, None, None, None, A_eq, INFEASIBLE

        # find improveable column
        index_in = c[:-1].argmax()

        targets = A_eq[:, index_in] > tol
        # compute b / A[:, in]
        ratio = np.full_like(A_eq[:, n], np.inf)
        ratio[targets] = A_eq[:, n][targets] / A_eq[targets, index_in]
        # find the row to push in
        final_target = ratio.argmin()
        # find the variable to turn out
        index_out = np.logical_and(base_row == final_target, base == 1).argmax()

        # pivot the base.
        base[index_out] = 0
        base[index_in] = 1
        base_row[index_in] = base_row[index_out]

        # change base
        A_eq = change_pivot_base(A_eq, final_target, index_in)

        # update c
        c -= c[index_in] * A_eq[final_target, :]


def finish(base, c, A_eq):
    # Check whether the LP solution finishes successfully (or ends with no solution)
    m, n = A_eq.shape
    A_eq = A_eq[:, :n - 1]
    c = c[:n - 1]
    # base component of c
    c_base = c[base == 0]
    # indices of variables that can be improved
    next_to_improve = (c_base > tol).nonzero()[0]

    if len(next_to_improve) == 0:
        # optimal
        return True, SUCCESS
    else:
        # check bounds
        max_col = A_eq.max(axis=0)
        if (max_col + tol < 0).sum() > 0:
            # if some variables can be infinitely improved, result is boundless
            return True, INFEASIBLE
        else:
            # return false for finishness
            return False, SUCCESS


def change_pivot_base(A_eq, row, col):
    A_eq[row, :] /= A_eq[row, col]
    not_zeroes = not_zero(A_eq[:, col])
    not_zeroes[row] = False
    index = not_zeroes.nonzero()[0]
    A_eq[index, :] -= A_eq[row] * A_eq[index, col:col + 1]

    return A_eq


def __simplex_stage1(A_eq, b_eq):
    m, n = A_eq.shape

    # if b is negative, twist it.
    b_eq[(b_eq < 0)] *= -1
    A_eq[(b_eq < 0), :] *= -1

    # slack variables
    A_eq = np.hstack((A_eq, np.eye(m)))
    # base variable indicator
    base = np.hstack((np.zeros(n, dtype=bool), np.ones(m, dtype=bool)))
    # base variable rows
    base_row = np.hstack([np.zeros(n, dtype=int), np.arange(m, dtype=int)])
    c = np.hstack((np.zeros(n), np.ones(m)))

    value, X, base, base_row, H, result = __simplex(-c, A_eq, b_eq, base, base_row)

    if result == INFEASIBLE:
        return INFEASIBLE, None, None, None, None
    else:
        if np.abs(value) > tol:
            return INFEASIBLE, None, None, None, None
        for i in range(n, n + m):
            if base[i] == 1:
                # row index
                row = base_row[i]
                for col in range(n):
                    # non zero column
                    if abs(H[row][col]) > tol:
                        base[col] = 1
                        base_row[col] = row
                        # change pivot.
                        H = change_pivot_base(H, row, col)
                        break

    A_eq = H[:, :n]
    b_eq = H[:, n + m]
    base_row = base_row[:n]
    base = base[:n]

    return result, base, base_row, b_eq, A_eq


# 改为求解最小化
def simplex(c, A_ub, b_ub, A_eq, b_eq, bounds):
    N_original = max(0 if A_eq is None else len(A_eq[0]), 0 if A_ub is None else len(A_ub[0]))
    c, A_eq, b_eq = __preprocess_bounds(c, A_eq, b_eq, A_ub, b_ub, bounds)
    # Stage1:
    # try to find a initial solution
    result, base, base_row, b_eq, A_eq = __simplex_stage1(A_eq, b_eq)
    if result == INFEASIBLE:
        return None, None, INFEASIBLE

    # Stage2: actually solve the problem
    value, X, _, _, _, result = __simplex(c, A_eq, b_eq, base, base_row)

    if result == INFEASIBLE:
        return value, X, INFEASIBLE
    elif result == SUCCESS:
        return -value, X[:N_original], SUCCESS


def __preprocess_bounds(c, A_eq, b_eq, A_ub, b_ub, bounds):
    # 保证A_eq, A_ub中每个row等长
    start = time.time()
    N_original = max(0 if A_eq is None else len(A_eq[0]), 0 if A_ub is None else len(A_ub[0]))
    # print(len(A_eq[0]))
    assert N_original > 0
    assert A_eq is None or all(len(row) == N_original for row in A_eq)
    assert A_ub is None or all(len(row) == N_original for row in A_ub)
    # 保证A_eq, ub分别和b_eq, ub等长
    assert (A_eq is None and b_eq is None or len(A_eq) == len(b_eq)) and (
                A_ub is None or b_ub is None or len(A_ub) == len(b_ub))
    # 保证lb和ub等长、合法
    assert bounds is None or len(bounds) == 0 or len(bounds) == N_original
    assert bounds is None or all(bound[0] <= bound[1] for bound in bounds)

    lb = None if bounds is None else [bound[0] for bound in bounds]
    ub = None if bounds is None else [bound[1] for bound in bounds]

    # all slack variables:
    if A_ub is None:
        A_ub = []
    if A_eq is None:
        A_eq = []
    if b_ub is None:
        b_ub = []
    if b_eq is None:
        b_eq = []
    unequation_slack_num = len(A_ub)
    bounded_var_slack_num = 0 if bounds is None else (sum(1 for x in lb if x != 0) + sum(1 for x in ub if x != INF))
    slack_total_num = unequation_slack_num + bounded_var_slack_num
    # xn after loosing A_ub, before loosing lb and ub
    N_after_slack_unequ = N_original + unequation_slack_num
    for i, row in enumerate(A_ub):
        row += [1 if j == i else 0 for j in range(unequation_slack_num)]

    for row in A_eq:
        row += [0] * unequation_slack_num

    A_eq += A_ub
    b_eq += b_ub
    c += [0] * slack_total_num
    for row in A_eq:
        row += [0] * bounded_var_slack_num
    # bounds
    # at this point, A_ub/b_ub has been merged into A_eq/b_eq, c has totally finished all the process (no need to care)
    # count how many bounds has been added.
    total_n = N_original + slack_total_num
    cur_xn = N_after_slack_unequ
    for i, bound in enumerate(lb):
        if bound > 0:
            # a specially bounded variable
            new_row = [0] * total_n
            new_row[i] = 1
            new_row[cur_xn] = -1
            A_eq.append(new_row)
            b_eq.append(bound)
            cur_xn += 1

    for i, bound in enumerate(ub):
        if bound < INF:
            # a specially bounded variable
            new_row = [0] * total_n
            new_row[i] = 1
            new_row[cur_xn] = 1
            A_eq.append(new_row)
            b_eq.append(bound)
            cur_xn += 1

    print("finish processing bounds.")
    print(time.time() - start)

    return -np.array(c, np.float64), np.array(A_eq, np.float64), np.array(b_eq, np.float64)


def testcase_2():
    c = [1, -14, 6]
    A_eq = None
    b_eq = None
    A_ub = [[-1, -1, -1],
            [0, 3, 1]]
    b_ub = [-4,
            6]
    # bounds = None
    bounds = [[1, 2], [0, 6], [0, 8]]
    return c, A_eq, b_eq, A_ub, b_ub, bounds


if __name__ == "__main__":
    N, c, A_ub, b_ub, A_eq, b_eq, bounds = process(get_case("TestCase3.txt"))

    start = time.time()
    v, X, success = simplex(c, A_eq, b_eq, A_ub, b_ub, bounds)
    end = time.time()
    print(end - start)
    print("State: %s" % success)
    print("X:")
    print(X)
    print("max value:")
    print(v)
    print(end - start)
