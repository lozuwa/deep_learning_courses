import numpy as np 

def reward_message():
	print('Tests passed')

def test_initialize_weights(x, initialize_weights):
	"""
	Unit test for the initialize_weights function
	"""
	weights = initialize_weights(x)
	assert x.shape[1] == weights.shape[0], 'The shape of the weights is not correct, remember weights is a column vector'
	reward_message()

def test_compute_hypothesis(compute_hypothesis):
	"""
	Unit test for the compute_hypothesis function
	"""
	features = np.array([1,2,3]).reshape(1, 3)
	weights = np.array([2,3,4]).reshape(3, 1)
	assert compute_hypothesis(features, weights) == 20, 'Implementation not ok, remember to use np.dot for matrix multiplication' 
	reward_message()

def test_cost_function(cost_function):
	"""
	Unit test for the cost_function function
	"""
	hyp = np.array([1,2,3]).reshape(3,1)
	y = np.array([1,2,3]).reshape(3,1)
	assert cost_function(hyp, y) == 0, 'Implementation not ok, check MSE'
	reward_message()