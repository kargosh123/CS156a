import numpy as np

# CS 156a - Fall Term 2018
# Neymika Jain
# Below is a rough outline of how I produced the PLA model used for questions
# 7 through 10. Numpy was used to handle vector operations.

# create random points
# create line from 2 points as target function

# To determine y : y - y1 = m(x -x1) -> y = m(x - x1) + y1

# determine sign using numpy.sign
# store points using numpy.array
# Perform dot product on points, x, and weights, w, using numpy.dot

# Determine missclassified points by comparing the sign of dot(w^t, x) to the actual 
# result determined by finding y If there are any missclassified points, 
# w_new = w_old + y_n*x_n

# Approximate probability by taking an extra 1000 points per run and using the
# converged PLA model on these points ("outside N"). Count the number of
# missclassfications, average these results over 1000 runs

# misclassification is a helper function for PLA. For each point, check to see
# that signs of PLA results and target function results match.
# If not, return False. If all points are correctly classified, return True.
def misclassification(points, weights, results):
	for i in range(len(points)):
		result = np.sign(np.dot(weights, points[i]))
		if (result != results[i]):
			return False
	return True


# PLA implements the perceptron learning algorithm. Starting with an assumed
# target function using two random points, a random number of training points
# are created and used to train the model, which assigns weights to points,
# which start as a zero vector every run. After a number of iterations, the
# PLA then converges and is used on a set of 1000 points outside of the
# training data set to determine the probability every run that the target
# function is not the hypothetical function produced by the PLA. Both the
# iteration and probability values are then returned as a tuple. 
def PLA(N):
	(x1,y1) = np.random.uniform(-1.0,1.0,2)
	(x2,y2) = np.random.uniform(-1.0,1.0,2)

	# Determine the parameters of target function f(x) = mx + b
	# using the two points randomly chosen above
	m = (y2-y1) / (x2-x1)
	b = y1 - m * x1

	# Create a list of random training points
	points = np.array([np.random.uniform(-1.0,1.0,2) for i in range(N)])
	points = np.insert(points, 0, 1, axis = 1)

	# According to the target function, determine the results to check
	# against what the PLA determines
	results = []
	for point in points:
		(x, y) = (point[1], point[2])
		f_x = m * x + b
		results.append(np.sign(y-f_x))

	# Initialize weight vector to a zero vector
	weights = np.zeros(3)
	iterations = 0

	# While there are still misclassiications, adjust the weights for the PLA
	while not misclassification(points, weights, results):
		random_point = np.random.randint(0,N)
		xn = points[random_point]
		yn = results[random_point]
		result = np.dot(weights, xn)
		if (np.sign(result) != yn):
			iterations += 1
			weights = np.add(weights, yn*xn)

	# Approximate the probability of f != g using a test set of 1000 points
	prob_points = np.array([np.random.uniform(-1.0,1.0,2) for i in range(1000)])
	prob_points = np.insert(prob_points, 0, 1, axis = 1)

	missclassified = 0.0
	for point in prob_points:
		(x, y) = (point[1], point[2])
		f_x = m * x + b
		if (np.sign(np.dot(weights, point)) != np.sign(y-f_x)):
			missclassified += 1

	miss_prob = missclassified/1000

	return (iterations, miss_prob)

def PLA_test(N):
	iterations, prob = 0.0, 0.0
	for i in range(1000):
		run_iterations, run_miss_prob = PLA(N)
		iterations += run_iterations
		prob += run_miss_prob

	avgIterations = iterations/1000
	avgProbability = prob / 1000
	print "Average number of Iterations:", avgIterations
	print "Probability of error: ", avgProbability

PLA_test(10)
PLA_test(100)