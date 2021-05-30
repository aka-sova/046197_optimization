import os, sys
import numpy as np
import matplotlib.pyplot as plt
import time


def analytical_estimation(X, y):
    # get the desired parameters with analytical estimation

    X_T_X = np.matmul(np.transpose(X), X)
    X_T_X_inv = np.linalg.inv(X_T_X)
    X_T_y = np.matmul(np.transpose(X),y)

    a_est = np.matmul(X_T_X_inv,X_T_y)

    return a_est


def projected_gd(X, y, step_size_mode : str, r, eps, max_iter, m, a_init, a_compare, eta = None, batch_size = None, print_every=100):
    """Estimate parameters using projected GD algorithm"""

    # step_size_mode :  decaying/AdaGrad


    def calc_grad(X, y, a, m, batch_size = None):

        if batch_size == None:
            X_a = np.matmul(X, a) # (m,)
            X_T = np.transpose(X) # (n x m)
            # y (m, 1)

            g_T = (1/m) * np.matmul(X_T, X_a-y[:,0])

        else:
            # choose random 'batch_size' number of examples
            # and calc gradient on them
            m, n = X.shape
            batch_indexes = np.random.randint(0, m, batch_size)
            X_batch = X[batch_indexes, :]
            y_batch = y[batch_indexes, :]

            X_a = np.matmul(X_batch, a)             # (batch_size,)
            X_batch_T = np.transpose(X_batch)       # (n x batch_size)
            # y (batch_size, 1)

            g_T = (1/m) * np.matmul(X_batch_T, X_a-y_batch[:,0])

        return g_T

    def get_G(X, y, r, m):

        mx = (np.matmul(np.transpose(X), X)) * (1/m)
        max_ev = np.max(np.linalg.eig(mx)[0])

        mx2 = (np.matmul(np.transpose(X), y)) * (1/m)
        norm = np.linalg.norm(mx2)

        G = 2*r*max_ev +  norm
        return G

    def calc_decaying_grad(D_G, t):
        step_size = D_G * (1/np.sqrt(t))
        return step_size

    def calc_adagrad(D, g_T_vec):

        norms_squared = np.linalg.norm(g_T_vec, axis=1)**2
        step_size = D / np.sqrt(2 * np.sum(norms_squared))
        return step_size

    def project_a(a, r):

        if np.linalg.norm(a) > r:
            return r * (a / np.linalg.norm(a))
        else:
            return a


    D = 2*r
    G = get_G(X, y, r, m)
    D_G = D/G

    a_compare = np.transpose(a_compare)

    # take the initial values as current estimation
    a_est = a_init

    # 2. keep going until stop condition reached or max_iter
    t = 1
    err_vec = np.array([])
    time_vec = np.array([])

    cur_grad = calc_grad(X, y, a_est, m, batch_size)
    cur_grad_expand = np.transpose(np.expand_dims(cur_grad, axis=1))
    grad_vec = cur_grad_expand

    start_time = time.time()

    while True:

        if (np.linalg.norm(cur_grad) ** 2 < eps):
            print(f"\nCurrent gradient {np.linalg.norm(cur_grad) ** 2} < {eps}. Stopping\n")
            break

        if (t > max_iter):
            print(f"\nReached max number of iterations: {max_iter}. Stopping\n")
            break

        # 3. calculate the step size
        if step_size_mode == 'decaying':
            step_size = calc_decaying_grad(D_G, t)
        elif step_size_mode == 'AdaGrad':
            step_size = calc_adagrad(D, grad_vec)
        elif step_size_mode == 'Const':
            step_size = eta
        else:
            raise Exception("Invalid step size mode")

        # 4. update the estimated params
        update_a = a_est - step_size * cur_grad
        projected_update_a = project_a(update_a, r)
        a_est = projected_update_a

        # 5. re-calculate the gradient
        cur_grad = calc_grad(X, y, a_est, m, batch_size)

        if step_size_mode == 'AdaGrad':
            cur_grad_expand = np.transpose(np.expand_dims(cur_grad, axis=1))
            grad_vec = np.concatenate((grad_vec, cur_grad_expand), axis=0)


        # 6. calc the error
        error = np.sum(np.abs(a_compare - a_est))
        err_vec = np.append(err_vec, error)

        time_vec = np.append(time_vec, time.time())

        if t%print_every == 0:
            print(f"Iter: {t}, error : {error}, grad_norm : {np.linalg.norm(cur_grad)**2}")


        t += 1


    return time_vec, err_vec, a_est

def main():

    # def params
    n = 3
    m = 10000
    a_vec = np.expand_dims(np.array([0, 1, 0.5, -2]), axis=1)

    x_min = -1
    x_max = 1

    mu, sigma = 0, np.sqrt(0.5) # mean and standard deviation

    x_vals = np.linspace(x_min, x_max, num=m)
    X_matrix = np.zeros([m, n+1])

    for i in range(n+1):
        X_matrix[:,i] = x_vals**i

    noise_vals = np.expand_dims(np.random.normal(mu, sigma, m), axis = 1)

    y_true = np.matmul(X_matrix, a_vec)
    y_meas = y_true + noise_vals



    # question 3 + 4
    a_est = analytical_estimation(X_matrix, y_meas)
    y_est = np.matmul(X_matrix, a_est)

    # plot all three graphs
    fig, ax = plt.subplots()
    ax.plot(x_vals, y_true, color='r')
    ax.plot(x_vals, y_est, color='g')
    ax.scatter(x_vals, y_meas, color='b', marker='o', s=0.1 )
    ax.legend(["True function f(x)", "Estimated function", "Measurements"], bbox_to_anchor=(0.55, 0.3))
    ax.set_title('True and Estimated function, and measurements')
    plt.savefig("graphs/q_3.png")
    plt.close(fig)

    do_56 = False
    do_78 = False
    do_10 = True
    do_11 = True

    # question 5 + 6

    # def params
    r = 4              # euclidean distance limit for vec a
    eps = 1e-10         # stop condition
    max_iter = 2000    # max steps

    # 1. pick random initial values s.t. they are in C:
    in_C = False

    while in_C == False:
        a_init = np.random.randn(4)
        in_C = np.linalg.norm(a_init) <= r


    if do_56:
        _, decay_err_vec, decay_a_est = projected_gd(X_matrix, y_meas, 'decaying', r, eps, max_iter, m, a_init, a_est)
        _, adagrad_err_vec, adagrad_a_est = projected_gd(X_matrix, y_meas, 'AdaGrad', r, eps, max_iter, m, a_init, a_est)


        # plot error
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.plot(decay_err_vec, color='r')
        ax.plot(adagrad_err_vec , color='b')
        ax.grid()
        ax.set_ylabel("Error [log]")
        ax.set_xlabel("Iterations")
        ax.legend(["Decaying step size", "Adagrad"], bbox_to_anchor=(0.55, 0.3))
        ax.set_title('Convergence rate comparison')
        plt.savefig("graphs/q_6.png")
        plt.close(fig)


    # Question 8 - 9
    mx = (np.matmul(np.transpose(X_matrix), X_matrix)) * (1 / m)
    L = np.max(np.linalg.eig(mx)[0])
    max_iter = 500    # max steps

    if do_78:

        _, const_step_1_err_vec, const_step_a_est_1 = projected_gd(X_matrix, y_meas, 'Const', r, eps, max_iter, m, a_init, a_est , eta=10/L)
        _, const_step_2_err_vec, const_step_a_est_2 = projected_gd(X_matrix, y_meas, 'Const', r, eps, max_iter, m, a_init, a_est , eta=1/L)
        _, const_step_3_err_vec, const_step_a_est_3 = projected_gd(X_matrix, y_meas, 'Const', r, eps, max_iter, m, a_init, a_est , eta=1/(10*L))

        # plot error
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.plot(const_step_1_err_vec, color='r')
        ax.plot(const_step_2_err_vec, color='b')
        ax.plot(const_step_3_err_vec, color='g')
        ax.grid()
        ax.set_ylabel("Error [log]")
        ax.set_xlabel("Iterations")
        ax.legend(["10/L", "1/L", "1/10L"], bbox_to_anchor=(0.55, 0.3))
        ax.set_title('Convergence rate comparison')
        plt.savefig("graphs/q_8.png")
        plt.close(fig)


        # plot error
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.plot(adagrad_err_vec, color='r')
        ax.plot(const_step_2_err_vec, color='b')
        ax.grid()
        ax.set_ylabel("Error [log]")
        ax.set_xlabel("Iterations")
        ax.legend(["Adagrad", "Const: eta=1/L"], bbox_to_anchor=(0.55, 0.3))
        ax.set_title('Convergence rate comparison')
        plt.savefig("graphs/q_9.png")
        plt.close(fig)


    # Quesitons 10 - 11

    batch_size_vec = [1, 10, 100, 1000]
    decay_err_vec_batch = {}
    decay_a_est_batch = {}

    eps = 1e-20         # stop condition
    max_iter = 1000    # max steps

    if do_10:
        for i, batch_size in enumerate(batch_size_vec):
            print(f"\nInit for batch_size = {batch_size}")
            _, decay_err_vec_batch[i], decay_a_est_batch[i] = projected_gd(X_matrix, y_meas, 'decaying', r, eps, max_iter, m, a_init, a_est, batch_size=batch_size, print_every=100)


        # plot error
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.plot(decay_err_vec_batch[0], color='r')
        ax.plot(decay_err_vec_batch[1], color='b')
        ax.plot(decay_err_vec_batch[2], color='g')
        ax.plot(decay_err_vec_batch[3], color='k')
        ax.grid()
        ax.set_ylabel("Error [log]")
        ax.set_xlabel("Iterations")
        ax.legend(["batch_size = 1", "batch_size = 10", "batch_size = 100", "batch_size = 1000"], bbox_to_anchor=(0.55, 0.3))
        ax.set_title('Convergence rate comparison')
        plt.savefig("graphs/q_10.png")

    decay_time_vec_batch = {}
    
    if do_11:

        max_iter = 10000  # max steps


        for i, batch_size in enumerate(batch_size_vec):
            print(f"\nInit for batch_size = {batch_size}")
            decay_time_vec_batch[i], decay_err_vec_batch[i], decay_a_est_batch[i] = projected_gd(X_matrix, y_meas, 'decaying', r, eps, max_iter, m, a_init, a_est, batch_size=batch_size, print_every=100)


        # plot error
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.plot(decay_time_vec_batch[0] - decay_time_vec_batch[0][0], decay_err_vec_batch[0], color='r')
        ax.plot(decay_time_vec_batch[1] - decay_time_vec_batch[1][0], decay_err_vec_batch[1], color='b')
        ax.plot(decay_time_vec_batch[2] - decay_time_vec_batch[2][0], decay_err_vec_batch[2], color='g')
        ax.plot(decay_time_vec_batch[3] - decay_time_vec_batch[3][0], decay_err_vec_batch[3], color='k')
        ax.grid()
        ax.set_ylabel("Error [log]")
        ax.set_xlabel("Time [s]")
        ax.legend(["batch_size = 1", "batch_size = 10", "batch_size = 100", "batch_size = 1000"], bbox_to_anchor=(0.55, 0.3))
        ax.set_title('Convergence rate comparison')
        plt.savefig("graphs/q_11.png")



    

    print("done")











if __name__ == "__main__":
    main()