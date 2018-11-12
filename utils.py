import tensorflow as tf

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def subpixel2d(x,shape):
	out = tf.depth_to_space(x,2)
	out = tf.reshape(out,shape=shape)
	return out

def batch_norm(x,training,name):
	return tf.layers.batch_normalization(x,training=training,name=name,reuse=tf.AUTO_REUSE)

def lrelu(x):
	return tf.nn.leaky_relu(x)

def prelu(input_tensor,scope_name):
	with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
		alpha = tf.get_variable('parameter',shape=input_tensor.get_shape()[-1],dtype=input_tensor.dtype,initializer=tf.constant_initializer(0.1),trainable=True)
		return tf.maximum(0.0,input_tensor) + alpha*tf.minimum(0.0,input_tensor)

def tanh(x):
	return tf.nn.tanh(x)

def variable(scope,var,shape)
	scope_name = str(scope)
	var_name = 'w_' + str(var)
	with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE):
		var = tf.get_variable(var_name,shape=shape,dtype=tf.float32,initializer=tf.glorot_uniform_initializer(),trainable=True)
	return var
