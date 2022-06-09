import numpy as np


def linear_approx_boxc_l2(x, v, c, verbose=True):
    """
    Solves the optimization problem:
    min norm(delta)_2
        s.th. <v, delta> = f_desired(x) - f_org(x)
        l <= delta + x <= u   (componentwise)
    INPUT:
        x is the input point which is about to be changed
        v = grad(f_org, x) - grad(f_desired, x) evaluated at x
        c = f_desired(x) - f_org(x), where
            f_desired is the desired class into which we want to change the classifier,
            f_org is the original class org=argmax_r f_r(x)
    OUTPUT:
        delta: change of sample (adversarial sample is: x + delta)
        opt_failed: True if problem not feasible, False if is solution found
    """

    d = len(x)
    if c >= 0:
        if verbose:
            print('There exists no feasible solution (c is already >= 0)')
        return np.zeros(d), True

    ixx = np.where(v != 0)[0]
    if len(ixx) == 0:
        if verbose:
            print('There exists no feasible solution (all grad diffs = 0)')
        return np.inf * np.ones(d), True

    x_sorted, v_sorted = x[ixx], v[ixx]
    lmbd_candidates = np.maximum(x_sorted / v_sorted, (x_sorted - 1) / v_sorted)
    print(f"lmbd_candidates: {lmbd_candidates}")
    icc = np.argsort(lmbd_candidates)
    lmbds_sorted = lmbd_candidates[icc]
    tot = np.sum(v ** 2)
    sumg, bdpart, counter = 0, 0, 0
    while sumg > c and counter < len(ixx):
        imm = ixx[icc[counter]]
        tot -= v[imm] ** 2
        if v[imm] > 0:
            bdpart += v[imm] * -x[imm]
        else:
            bdpart += v[imm] * (1 - x[imm])
        sumg = -lmbds_sorted[counter] * tot + bdpart
        counter += 1
    if sumg <= c:
        tot += v[imm] ** 2
        if v[imm] > 0:
            bdpart -= v[imm] * -x[imm]
        else:
            bdpart -= v[imm] * (1 - x[imm])
        print(f"bdpart:{bdpart}; imm:{imm}; tot:{tot}")
        lmbd_opt = (bdpart - c) / tot
        if lmbd_opt > lmbds_sorted[counter - 1]:
            if verbose:
                print('There exists no feasible solution (lmbd > sl(counter-1))')
            return np.zeros(d), True
    else:
        if verbose:
            print('There exists no feasible solution')
        return np.zeros(d), True

    delta = np.maximum(-x, np.minimum(-lmbd_opt * v, 1 - x))
    if verbose:
        print(f"x           : {x}")
        print(f"v           : {v}")
        print(f"c           : {c}")
        print(f"lmbd_opt    : {lmbd_opt}")
        print(f"delta       : {delta}")
        print(f"delta.dot(v): {delta.dot(v)}")
        print(np.linalg.norm(delta))
        print(-lmbd_opt * tot + bdpart)

    return delta, False


# linear_approx_boxc_l2(np.array([0.3, 0.75, 0.8, 0.9]), np.array([3.0, 1.5, 1.0, 1.0]), -1.51)
# x           : [0.3    0.75    0.8     0.9 ]
# v           : [3.     1.5     1.      1. ]
# c           : -1.51
# lmbd_candidates: [0.1     0.5     0.8     0.9]
# lmbd_opt       : 0.14353    <<< it is between 0.1 and 0.5
# bdpart      :-0.9
# imm         :1
# tot         :4.25
# delta       : [-0.3        -0.21529412    -0.14352941     -0.14352941]
# delta.dot(v): -1.51
# np.linalg.norm(delta)   : 0.42137
# -lmbd_opt * tot + bdpart: -1.51

# results = pool.map(linear_approx_boxc_l2, zip((x, v, c), (x, v, c)))
# results = pool.starmap(linear_approx_boxc_l2, [(x, v, c), (x, v, c)])


# import time
# d = 3000
# l = np.zeros(d)
# u = np.ones(d)
# start = time.time()
# for i in range(5000):
#     x = np.random.rand(d)
#     v = np.random.rand(d) * 2 - 1
#     c = -np.random.rand()*20
#     delta1 = linear_approx_boxc_l2(x, v, c)
# print("total time:", time.time() - start)


# import time
# from multiprocessing import Pool
# n_threads = 9
# d = 3000
# pool = Pool(n_threads)
# start = time.time()
# x = np.random.rand(n_threads, d)
# v = np.random.rand(n_threads, d) * 2 - 1
# c = -np.random.rand(n_threads)*20
# for j in range(640//n_threads):
#     results = pool.starmap(linear_approx_boxc_l2, zip(x, v, c))
#     # delta1 = linear_approx_boxc_l2(x[0], v[0], c[0])
# pool.close()
# pool.join()
# print("total time:", time.time() - start)

