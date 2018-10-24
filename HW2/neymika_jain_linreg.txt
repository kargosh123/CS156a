import numpy as np
import pla

# CS 156a - Fall Term 2018
# Neymika Jain
# Below is a rough outline of how I produced the linear regression model used for questions
# 5 through 7. Numpy was used to handle vector operations.
# Linear Regression:
# Generate the target function, using two random points
# Create a training set of points with corresponding outputs
# Train the model by using the formula for change in w given in Lecture 3:
# X_cross = (X^T X)^-1 X^T
# Determine the in-sample error, using squared error
# Test:
# For each run, determine the in-sample error, out-of-sample error, and 
# pocket model iteration convergence (starting with lin reg weights)
# Average these three quantities and print

class linreg:        
    def __init__(self, numpoints, d):
        self.initp = [np.random.uniform(-1.0,1.0,2) for x in range(2)]
        self.gen_points(numpoints)
        (x_diff, y_diff) = np.subtract(self.initp[1], self.initp[0])
        self.target_slope = y_diff/x_diff
        self.target_b = self.initp[0][1] - (self.target_slope * self.initp[0][1])
        self.Y = np.array([np.sign(x - self.initp[0][0]) for x in self.X])
        self.weights = np.zeros((1+d,1))

    def gen_points(self, numpoints):
        self.n = numpoints
        self.X = np.random.uniform(-1.0,1.0,(self.n, 2))
        self.Y = np.array([np.sign(x - self.initp[0][0]) for x in self.X])

    def train(self):
        X_cross = np.c_[np.ones(self.X.shape[0]), self.X]
        X_inv = np.linalg.pinv(X_cross)
        self.weights = np.dot(X_inv,self.Y)
        
    def e_in(self):
        xw = np.c_[np.ones(self.X.shape[0]), self.X].dot(self.weights)
        xw = np.sign(xw)
        return np.mean(np.not_equal(xw, self.Y))
        
def linreg_test(runs):
    N_1, N_2, N_3, d = 100, 1000, 10, 2
    ein, eout, iterations = [], [], []
    for i in range(runs):
        lr = linreg(N_1, 2)
        lr.train()
        ein.append(lr.e_in())
        lr.gen_points(N_2)
        eout.append(lr.e_in())
        iterations.append(pla.PLA_test(N_3, 1, lr.weights, lr.target_slope, lr.target_b))
    print("e_in average: %f" % np.average(np.array(ein)))
    print("e_out average: %f" % np.average(np.array(eout)))
    print("perceptron convergence average: %f" % np.average(np.array(iterations)))

linreg_test(1000)