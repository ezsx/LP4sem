import time
import sys
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
start_time = time.perf_counter()


def f(x):
    return x * x * x - 18 * x - 83


def df(x):
    return 3 * x * x - 18


def _rectangle_rule(func, a, b, num_of_iter, mult):
    step = (b - a) / num_of_iter
    result_sum = 0.0
    x_start = a + mult * step
    for i in range(1, int(num_of_iter)):
        result_sum += func(x_start + i * step)
    return result_sum * step


def left_rectangle_rule(func, a, b, n):
    return _rectangle_rule(func, a, b, n, 0.0)


def newton(start, end, eps, a, res, num_of_iter):
    while abs(end - start) >= eps:
        local_num_of_iter = num_of_iter / p
        h_func = (end - a) / num_of_iter
        local_a = a + my_rank * local_num_of_iter * h_func
        local_b = local_a + local_num_of_iter * h_func
        fx = left_rectangle_rule(f, local_a, local_b, local_num_of_iter)
        dfx = left_rectangle_rule(df, local_a, local_b, local_num_of_iter)
        if my_rank == 0:
            fx1, fx2 = fx, dfx
            for source in range(1, p):
                integral_array = comm.recv(source=ANY_SOURCE)
                fx1, fx2 = fx1 + integral_array[0], fx2 + integral_array[1]
            fx1, fx2 = fx1 - res, fx2 - res
            start = end
            end -= fx1 / fx2
            for rank_index in range(1, p):
                comm.send([start, end], dest=rank_index)
        else:
            comm.send([fx, dfx], dest=0)
            segment_array = comm.recv(source=ANY_SOURCE)
            start, end = segment_array[0], segment_array[1]
    return end


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    my_rank, p = comm.Get_rank(), comm.Get_size()
    integral_from, integral_result, amount_of_iterations, epsilon, x0 = \
        float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    x1 = x0 + 0.5
    result = newton(x0, x1, epsilon, integral_from, integral_result, amount_of_iterations)
    end_time = time.perf_counter()
    if my_rank == 0:
        # example to run: mpiexec -np 4 python newton2.py 8.0 10.0 240000 0.00001 0.0
        print(f"Result: {result}")
        print(f"Time: {end_time - start_time}")
    MPI.Finalize()