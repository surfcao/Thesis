import tensorflow as tf
from utils import conv2d,prelu,lrelu,batch_norm,tanh,subpixel2d

class generator:
	def __init__(self,weights,biases,scope_name,shape):
		self.weights = weights
		self.biases = biases
		self.shape = shape
		self.scope_name = scope_name

	def gen(self,x,training):
		x = tf.reshape(x,shape=[-1,self.shape,self.shape,3])
		scope = str(self.scope_name)+'_'

		layer_1 = conv2d(x,weights[scope+'w_conv1']) + biases[scope+'b_conv1']
		layer_1 = lrelu(layer_1)

		layer_2 = conv2d(layer_1,weights[scope+'w_conv2']) + biases[scope+'b_conv2']
		layer_2 = batch_norm(layer_2,training=training,name=scope+'conv2')
		layer_2 = prelu(layer_2,scope+'w_conv2')

		layer_3 = conv2d(layer_2,weights[scope+'w_conv3']) + biases[scope+'b_conv3']
		layer_3 = batch_norm(layer_3,training=training,name=scope+'conv3')
		layer_3 = prelu(layer_3,scope+'w_conv3')

		layer_4 = conv2d(layer_3,weights[scope+'w_conv4']) + biases[scope+'b_conv4']
		layer_4 = batch_norm(layer_4,training=training,name=scope+'conv4')
		layer_4 = prelu(layer_4,scope+'w_conv4') + layer_3

		if self.scope_name==1:
			layer_4 = prelu(layer_4,scope+'w_conv4_res') + layer_2
			layer_5 = conv2d(layer_4,weights[scope+'w_conv5']) + biases[scope+'b_conv5']
			layer_5 = subpixel2d(layer_5,[-1,self.shape*2,self.shape*2,64])
			layer_5 = prelu(layer_5,scope+'w_conv5')
			layer = layer_5

		elif self.scope_name==2:
			layer_5 = conv2d(layer_4,weights[scope+'w_conv5']) + biases[scope+'b_conv5']
			layer_5 = batch_norm(layer_5,training=training,name=scope+'conv5')
			layer_5 = prelu(layer_5,scope+'w_conv5') + layer_4 

			layer_6 = conv2d(layer_5,weights[scope+'w_conv6']) + biases[scope+'b_conv6']
			layer_6 = batch_norm(layer_6,training=training,name=scope+'conv6')
			layer_6 = prelu(layer_6,scope+'w_conv6') + layer_5
			layer_6 = prelu(layer_6,scope+'w_conv6_res') + layer_2

			layer_7 = conv2d(layer_6,weights[scope+'w_conv7']) + biases[scope+'b_conv7']
			layer_7 = subpixel2d(layer_7,[-1,self.shape*2,self.shape*2,64])
			layer_7 = prelu(layer_7,scope+'w_conv7')
			layer = layer_7

		elif self.scope_name==3:
			layer_5 = conv2d(layer_4,weights[scope+'w_conv5']) + biases[scope+'b_conv5']
			layer_5 = batch_norm(layer_5,training=training,name=scope+'conv5')
			layer_5 = prelu(layer_5,scope+'w_conv5') + layer_4 

			layer_6 = conv2d(layer_5,weights[scope+'w_conv6']) + biases[scope+'b_conv6']
			layer_6 = batch_norm(layer_6,training=training,name=scope+'conv6')
			layer_6 = prelu(layer_6,scope+'w_conv6') + layer_5 

			layer_7 = conv2d(layer_6,weights[scope+'w_conv7']) + biases[scope+'b_conv7']
			layer_7 = batch_norm(layer_7,training=training,name=scope+'conv7')
			layer_7 = prelu(layer_7,scope+'w_conv7') + layer_6 

			layer_8 = conv2d(layer_7,weights[scope+'w_conv8']) + biases[scope+'b_conv8']
			layer_8 = batch_norm(layer_8,training=training,name=scope+'conv8')
			layer_8 = prelu(layer_8,scope+'w_conv8') + layer_7
			layer_8 = prelu(layer_8,scope+'w_conv8_res') + layer_2

			layer_9 = conv2d(layer_8,weights[scope+'w_conv9']) + biases[scope+'b_conv9']
			layer_9 = subpixel2d(layer_9,[-1,self.shape*2,self.shape*2,64])
			layer_9 = prelu(layer_9,scope+'w_conv9')
			layer = layer_9

		output = conv2d(layer,weights[scope+'out']) + biases[scope+'out']
		output = 127.5*(1.0 + tanh(output))
		return output
