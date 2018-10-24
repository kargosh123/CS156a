from neymika_jain_linreg import linreg
import numpy as np

class NLTtest():
    def __init__(self, numpoints, noise, coeffs):
        self.n = numpoints
        self.points = np.random.uniform(-1.0,1.0,(self.n, 2))
        self.noise = max(min(1, noise), 0)
        self.target = NLTarget(coeffs)
        self.weights = np.zeros((1+2,1))
        self.nltweights = np.zeros((1+2, 1))
        self.nlt = LinRegNLT(2)
        self.coeffs = coeffs
        self.points_transformed = np.multiply(self.coeffs, np.c_[np.ones(self.points.shape[0]), np.square(self.points)])
        self.labels = np.sign(np.sum(self.points_transformed, axis=1))
        self.add_noise()

    def add_noise(self):
        amt = self.noise
        # number of indices to flip labels
        n_flip = int(self.n * amt)
        # label-flipping array consisting of
        # 1: don't flip
        # -1: flip
        flip_elts = np.multiply(-1, np.ones(n_flip))
        flip_arr = np.r_[np.ones(self.n - n_flip), flip_elts]
        np.random.shuffle(flip_arr)
        self.noisy_labels = np.multiply(flip_arr, self.labels)

    def regen_points(self, numpoints, amt):
        self.n = numpoints
        self.noise = max(min(1, amt), 0)
        self.points = np.random.uniform(-1.0,1.0,(self.n, 2))
        self.points_transformed = np.multiply(self.coeffs, np.c_[np.ones(self.points.shape[0]), np.square(self.points)])
        self.labels = np.sign(np.sum(self.points_transformed, axis=1))
        self.add_noise()

    def train(self):
        real_X = np.c_[np.ones(self.points.shape[0]), self.points]
        pinv_X = np.linalg.pinv(real_X)
        self.weights = np.dot(pinv_X,self.labels)

        real_X = np.c_[np.ones(self.points.shape[0]), self.points, np.prod(self.points, axis=1), np.square(self.points)]
        pinv_X = np.linalg.pinv(real_X)
        self.nltweights = np.dot(pinv_X,self.noisy_labels)
        
    def e_in(self):
        xw = np.c_[np.ones(self.points.shape[0]), self.points].dot(self.nltweights)
        self.nlt_e_in = np.not_equal(np.sign(xw), self.noisy_labels).mean()

        xw = np.c_[np.ones(self.points.shape[0]), self.points].dot(self.weights)
        self.lr_e_in = np.not_equal(np.sign(xw), self.noisy_labels).mean()
        

def prob(num_exp):
    numpts = 1000
    noise_amt = 0.1
    coeffs = np.array([-0.6, 1, 1])
    numweights = 100
    lr_ein = np.array([])
    nltlr_w = np.array([])
    nltlr_eout = np.array([])
    cur_nlt = NLTtest(numpts, noise_amt, coeffs)
    for i in range(num_exp):
        cur_nlt.regen_points(numpts, noise_amt)
        cur_nlt.train()
        cur_nlt.e_in()
        cur_lrein = cur_nlt.lr_ein
        lr_ein = np.concatenate((lr_ein,[cur_lrein]))
        cur_nltlrw = cur_nlt.nltweights
        if i < numweights:
            nltlr_w = np.concatenate((nltlr_w,cur_nltlrw))
        cur_nlt.regen_points(numpts, noise_amt)
        cur_nltlreout = cur_nlt.nlt_e_in
        nltlr_eout = np.concatenate((nltlr_eout,[cur_nltlreout]))

    nltlr_w = nltlr_w.reshape(numweights, 6)
    avg_lr_ein = np.average(lr_ein)
    avg_nlt_w = np.average(nltlr_w, axis=0)
    avg_nlt_eout = np.average(nltlr_eout)
    print("average linreg e_in: %f" % avg_lr_ein)
    print("average nonlinreg weights:")
    print(avg_nlt_w)
    print("average nonlinreg e_out: %f" % avg_nlt_eout)
prob(1000)