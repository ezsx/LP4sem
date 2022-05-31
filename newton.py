from mpi4py import MPI
import math
import time
import sys
start_time = time.perf_counter()
def df(x):
    return x * x-3*x+5
def rectangle_rule_right(func, a, b, num_of_iter):
    step = (b - a) / num_of_iter
    result_sum = 0.0
    x_start = a+1.0*step
    for i in range(1, int(num_of_iter)):
        result_sum += func(x_start + i * step)
    return result_sum * step
def newton(a, b, x0, comm, eps, kmax, total_num_iter):
    my_rank = comm.Get_rank()
    if my_rank == 0:
        p = comm.Get_size()
        x = x0
        x_prev, i = x0 + 2 * eps, 0
        while abs(x - x_prev) >= eps and i < kmax:
            ab = (x - a) / (p - 1)
            for j in range(1, p):
                d = {'a': a + (j - 1) * ab,
                     'b': a + j * ab,
                     'num_of_iter': total_num_iter / (p - 1),
                     'is_stop': False}
                comm.send(d, dest=j)
            integr_x = 0
            for j in range(1, p):
                integr_x += comm.recv(source=j)
            x_prev = x
            x = x - (integr_x - b) / df(x)
            print(x, ' : ', x_prev)
            i += 1
        for j in range(1, p):
            d = {'is_stop': True}
            comm.send(d, dest=j)
        return x
    else:
        while True:
            d = comm.recv(source=0)
            if d['is_stop']:
                break
            else:
                res = rectangle_rule_right(df, d['a'], d['b'], d['num_of_iter'])
                comm.send(res, dest=0)
def main():
    comm = MPI.COMM_WORLD
    a, b,  total_num_iter, epsilon, x0 = \
        float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    res = newton(a=a, b=b, x0=0, eps=epsilon, kmax=100, comm=comm, total_num_iter=total_num_iter)
    if comm.Get_rank() == 0:
        print('Result ! =>', res)
        end_time = time.perf_counter()
        print('Time   ! =>', end_time - start_time)
    MPI.Finalize
main()