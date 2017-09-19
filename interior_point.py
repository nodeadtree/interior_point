import numpy as np
# Optimizes a given linear function with linear non-negative constraints
# Using the barrier method provided by Calafiore and El Ghaoui's
# Optimization textbook
# Built for python 3+
# Works on python2.7

class Barrier_Method(object):

    # Initializes Barrier_Method object, with constraints, parameters,  and 
    # objective function
    #
    # Expected format for arguments is as follows
    # func                    - 1*n numpy matrix, the objective function
    #
    # constraint_coefficients - m*n numpy matrix, these are the coefficients
    #                           for the constraints
    #
    # constraints             - 1*m numpy matrix, these are the bounds 
    #                           for each of the constraints
    #
    # scaling_coefficient     - integer, This is how quickly the weight for 
    #                           the objective function grows
    #
    # intial_weight           - this is the starting weight for the objective 
    #                           function
    #
    # s_init                  - starting step size for backtracking linesearch
    #
    # beta                    - stepsize scaling in the backtracking 
    #                           linesearch
    #
    # alpha                   - threshold scalar for backtracking linesearch
    # 
    # epsilon                 - chosen threshold of smallness
    #                          

    def __init__(self, func, constraint_coefficients, constraints, 
            scaling_coefficient = 2, initial_weight = 1,
            s_init = 1, beta = .8, alpha = .9, epsilon = 10 ** -7):

        self.func = func
        self.constraint_coefficients = constraint_coefficients
        self.constraints = constraints
        self.scaling_coefficient = scaling_coefficient
        self.weight = initial_weight
        self.current_guess = self.choose_start()
        self.m, self.n = constraints.shape

        #Line search parameters and threshold value
        self.s_init = s_init
        self.beta = beta
        self.alpha = alpha
        self.epsilon = epsilon
        self.k = 0

    # Starts optimization process, using the logarithmic barrier function method
    #
    # visible                 - Boolean used to determine whether the user
    #                           wants a printed output or silent output.
    #                           By default, it is silent.
    def begin_optimization(self, visible=False):
        while self.m / float(self.weight) >= self.epsilon:
            self.k += 1
            self.newton()
            self.weight *= self.scaling_coefficient
            if visible:
                print('New x:' + str(self.current_guess))
                print('New value at x:' + str(self.objective(
                    self.current_guess.T)))
                print(self.k)
                print('threshold :' + str(self.m/float(self.weight)))
        return self.current_guess

    # Chooses starting point within the feasible region
    def choose_start(self):
        p = np.max(self.constraints) * np.matrix(np.ones(self.func.shape[1]))
        j = [False]
        while False in j:
            axb = self.constraints.T - self.constraint_coefficients * p.T
            j = [k > 0 for k in axb]
            p = 1 / float(2) * p if False in j else p
        return p

    # Uses the damped newton method to find the optimizer for a particular
    # weight of the barrier function
    #
    # max_runs                -integer, used to set a hard limit to the number
    #                          of iterations used in newton's method
    def newton(self, max_runs=200):
        weight = self.weight
        A = self.constraint_coefficients
        b = self.constraints.T
        c = self.func.T
        guess = self.current_guess.T
        x = guess
        i = 0
        while i < max_runs:
            func = self.objective(guess).T
            delta = 1 / (b - A * x)
            gradient = weight * c + A.T * delta
            hessian = A.T * np.diag(delta.A1) * np.diag(delta.A1) * A
            direction = np.linalg.solve(-hessian, gradient)
            lambda_k = -gradient.T * direction
            if lambda_k < self.epsilon:
                break
            else:
                t = self.s_init
                # Perform backtracking line search to choose a suitably small s
                while self.objective(x+t*direction) > (
                    func + self.alpha * t * gradient.T * direction):
                    t = self.beta * t
                    if t < self.epsilon:
                        break
                x = x + direction
            i += 1
        self.current_guess = x.T

    # Calculates the objective function for a given x
    def objective(self, x):
        ret = (self.weight * self.func) * x - np.sum(
                np.log((self.constraints.T - self.constraint_coefficients * x)))
        return ret


# Boilerplate and test case
if __name__ == '__main__':
    objective_function = np.matrix([1,1,1])
    constraint_coefficients = np.matrix([[-5,.5,0],
                                        [3,-2,0],
                                        [.4,.6,0],
                                        [-1,-2,0],
                                        [0,0,1],
                                        [0,0,-1]])

    constraints = np.matrix([1,2,3,-4,3,-1])
    optimizer = Barrier_Method(objective_function, constraint_coefficients,
                               constraints)
    print(optimizer.begin_optimization(True))
