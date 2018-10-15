from numpy import array, zeros, argmin, inf, ndim
from scipy.spatial.distance import cdist
import threading


def dtw_improved(x, y, dist, warp=1, l=0.3):
    r, c = len(x), len(y)
    lc = int(round(c * l))
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D = D0[1:, 1:]  # view
    D[0:, 0:] = inf

    t1 = threading.Thread(target=count_lines, args=(0, 3, x, y, dist, l, D))
    t2 = threading.Thread(target=count_lines, args=(1, 3, x, y, dist, l, D))
    t3 = threading.Thread(target=count_lines, args=(2, 3, x, y, dist, l, D))
    # t4 = threading.Thread(target=count_lines, args=(3, 5, x, y, dist, l, D))
    # t5 = threading.Thread(target=count_lines, args=(4, 5, x, y, dist, l, D))

    t1.start()
    t2.start()
    t3.start()
    # t4.start()
    # t5.start()

    t1.join()
    t2.join()
    t3.join()
    # t4.join()
    # t5.join()


    # for i in range(r):
    #     for j in range(max(i - lc, 0), min(i + lc, c)):
    #         #             if (c >= r - lc and c <= r + lc):
    #         D[i, j] = dist(x[i], y[j])
    # #             else:
    # #                 D1[i, j] = inf
    C = D.copy()
    for i in range(r):
        for j in range(max(i - lc, 0), min(i + lc, c)):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r - 1)
                j_k = min(j + k, c - 1)
                min_list += [D0[i_k, j], D0[i, j_k]]
            D[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D[-1, -1] / sum(D.shape), C, D, path


def count_lines(freq, thread_num, x, y, dist, l, D):
    r, c = len(x), len(y)
    lc = int(round(c * l))
    for i in range(freq, r, thread_num):
        for j in range(max(i - lc, 0), min(i + lc, c)):
            D[i, j] = dist(x[i], y[j])


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
