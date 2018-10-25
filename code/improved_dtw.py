from numpy import array, zeros, argmin, inf, ndim
from scipy.spatial.distance import cdist
import threading
from scipy.stats.mstats import zscore
import copy


def dtw_improved(x, y, dist, warp=1, l=0.3, zscr=False):
    if zscr:
        zscore(x)
        zscore(y)
    r, c = len(x), len(y)
    lc = int(round(c * l))
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    # D0[0, 0] = 0
    D = D0[1:, 1:]  # view
    D[0:, 0:] = inf

    a1, a2 = 0, 0

    for i in range(r + c - 1):
        t1 = threading.Thread(target=count_lines, args=(0, 3, x, y, copy.copy(a1), copy.copy(a2), dist, l, D, D0, warp))
        t2 = threading.Thread(target=count_lines, args=(1, 3, x, y, copy.copy(a1), copy.copy(a2), dist, l, D, D0, warp))
        t3 = threading.Thread(target=count_lines, args=(2, 3, x, y, copy.copy(a1), copy.copy(a2), dist, l, D, D0, warp))

        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()

        a1 = min(a1 + 1, r - 1)
        a2 = max(0, a1 - lc) + max((i + 2) - r, 0)




    # for i in range(r):
    #     for j in range(max(i - lc, 0), min(i + lc, c)):
    #         #             if (c >= r - lc and c <= r + lc):
    #         D[i, j] = dist(x[i], y[j])
    # #             else:
    # #                 D1[i, j] = inf
    # print(D0)
    # print("-----")
    # print(D)
    C = D.copy()
    # for i in range(r):
    #     for j in range(max(i - lc, 0), min(i + lc, c)):
    #         min_list = [D0[i, j]]
    #         for k in range(1, warp + 1):
    #             i_k = min(i + k, r - 1)
    #             j_k = min(j + k, c - 1)
    #             min_list += [D0[i_k, j], D0[i, j_k]]
    #         D[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)

    return D[-1, -1] / sum(D.shape), C, D, path


def count_lines(freq, thread_num, x, y, a1, a2, dist, l, D, D0, warp):
    r, c = len(x), len(y)
    lc = int(round(c * l))
    while a1 > 0 and D[a1 - 1, a2] == inf:
        print("ede")
        p = 1

    i = 0
    while True:
        a1 -= i * thread_num + freq
        a2 += i * thread_num + freq

        if a1 < 0 or a2 < max(a1 - lc, 0) or a2 > min(a1 + lc, c - 1):
            break
        if a1 == 0 and a2 == 0:
            min_list = [0]
        else:
            min_list = [D0[a1, a2]]
        for k in range(1, warp + 1):
            i_k = max(min(a1 + k, r - 1), 1)
            j_k = max(min(a2 + k, c - 1), 1)
            min_list += [D0[i_k, a2], D0[a1, j_k]]
        if min(min_list) == inf:
            min_list = [0]
        D[a1, a2] = min(min_list) + dist(x[a1], y[a2])
        if D[a1, a2] == inf:
            print(min_list)
        i += 1



    # for i in range(freq, r, thread_num):
    #     for j in range(max(i - lc, 0), min(i + lc, c)):
    #         D[i, j] = dist(x[i], y[j])


# def writer(x, event_for_wait, event_for_set):
#     for i in xrange(10):
#         event_for_wait.wait() # wait for event
#         event_for_wait.clear() # clean event for future
#         print x
#         event_for_set.set() # set event for neighbor thread
#
# # init events
# e1 = threading.Event()
# e2 = threading.Event()
# e3 = threading.Event()
#
# # init threads
# t1 = threading.Thread(target=writer, args=(0, e1, e2))
# t2 = threading.Thread(target=writer, args=(1, e2, e3))
# t3 = threading.Thread(target=writer, args=(2, e3, e1))
#
# # start threads
# t1.start()
# t2.start()
# t3.start()
#
# e1.set() # initiate the first event
#
# # join threads to the main thread
# t1.join()
# t2.join()
# t3.join()





def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)
