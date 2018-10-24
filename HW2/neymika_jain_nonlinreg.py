import numpy as np

# CS 156a - Fall Term 2018
# Neymika Jain
# Below is a rough outline of how I produced the linear regression model used for questions
# 5 through 7. Numpy was used to handle vector operations.
# NonLinear Regression:
# Generate the target function, using two random points that are transformed
# Create a training set of tranformed points with corresponding outputs
# Add noise to correspoinding output using deterministic function given in Lecture 4
# Train the lin reg model by using the formula for change in w given in Lecture 3:
# X_cross = (X^T X)^-1 X^T
# Train the non lin reg model using transformed points instead
# Determine the in-sample error, using squared error
# Test:
# For each run, determine the in-sample error for the linear regression model,
# out-of-sample error for nonlinear regression model, and the weights
# associated with the nonlinear regression model with noisy outputs
# Average these three quantities and print
class nonlinreg():
    def __init__(self, numpoints, d, noise, coeffs):
        self.weights = np.zeros((1+d,1))
        self.nltweights = np.zeros((1+d, 1))
        self.coeffs = coeffs
        self.gen_points(numpoints, noise)
        self.add_noise()

    def add_noise(self):
        n_flip = int(self.n * self.noise)
        flip_Y = np.r_[np.ones(self.n - n_flip), np.multiply(-1, np.ones(n_flip))]
        np.random.shuffle(flip_Y)
        self.noisy_labels = np.multiply(flip_Y, self.Y)

    def gen_points(self, numpoints, amt):
        self.n = numpoints
        self.noise = max(min(1, amt), 0)
        self.X = np.random.uniform(-1.0,1.0,(self.n, 2))
        self.X_T = np.multiply(self.coeffs, np.c_[np.ones(self.X.shape[0]), np.square(self.X)])
        self.Y = np.sign(np.sum(self.X_T, axis=1))
        self.add_noise()

    def train(self):
        X_cross = np.c_[np.ones(self.X.shape[0]), self.X]
        X_inv = np.linalg.pinv(X_cross)
        self.weights = np.dot(X_inv,self.Y)

        X_cross = np.c_[np.ones(self.X.shape[0]), self.X, np.prod(self.X, axis=1), np.square(self.X)]
        X_inv = np.linalg.pinv(X_cross)
        self.nltweights = np.dot(X_inv,self.Y)
        
    def e_in(self):
        xw = np.c_[np.ones(self.X.shape[0]), self.X, np.prod(self.X, axis=1), np.square(self.X)].dot(self.nltweights)
        self.nlt_e_in = np.not_equal(np.sign(xw), self.noisy_labels).mean()

        xw = np.c_[np.ones(self.X.shape[0]), self.X].dot(self.weights)
        self.lr_e_in = np.not_equal(np.sign(xw), self.noisy_labels).mean()
        

def nonlinreg_test(runs):
    numpoints, noise, d = 1000, 0.1, 2
    coeffs = np.array([-0.6, 1, 1])
    ein, w, eout = [], [], []
    for i in range(runs):
        nlt = nonlinreg(numpoints, d, noise, coeffs)
        nlt.train()
        nlt.e_in()
        ein.append(nlt.lr_e_in)
        w.append(nlt.nltweights)
        nlt.gen_points(numpoints, noise)
        nlt.e_in()
        eout.append(nlt.nlt_e_in)

    w = np.array(w)
    w = w.reshape(w.shape[0], 6)
    print("average linreg e_in: %f" % np.average(ein))
    print("average nonlinreg weights:")
    print(np.average(w, axis=0))
    print("average nonlinreg e_out: %f" % np.average(eout))

nonlinreg_test(1000)