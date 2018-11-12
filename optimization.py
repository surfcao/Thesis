import tensorflow as tf

tfco = tf.contrib.constrained_optimization

class constrained_problem(tfco.ConstrainedMinimizationProblem):
	def __init__(self,log_loss,vgg_loss,vgg_bound):
		self._objective = log_loss
		self._constraints = vgg_loss - vgg_bound

	@property
	def objective(self):
		return self._objective
	
	@property
	def constraints(self):
		return self._constraints
	
	
	