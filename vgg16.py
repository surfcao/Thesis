import tensorflow as tf
import numpy as np


vgg_mean_bgr = tf.constant([103.939,116.779,123.68],dtype=tf.float32)

def conv_layer(x,w,b,max_pool):
		conv = tf.nn.conv2d(x,w,[1,1,1,1],padding="SAME")	
		conv = tf.nn.bias_add(conv,b)
		conv = tf.nn.relu(conv)
		if max_pool:
			conv = tf.nn.max_pool(conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
		return conv

class vgg:
	
	def __init__(self,weight_file,sess,shape=128,layer_no=4):
		self.layer_no = layer_no
		self.shape = shape
		self.w = {
			#1
			'w_1':tf.Variable(tf.random_normal([3,3,3,64]),trainable=False),
			'w_2':tf.Variable(tf.random_normal([3,3,64,64]),trainable=False),
			#2
			'w_3':tf.Variable(tf.random_normal([3,3,64,128]),trainable=False),
			'w_4':tf.Variable(tf.random_normal([3,3,128,128]),trainable=False),
			#3
			'w_5':tf.Variable(tf.random_normal([3,3,128,256]),trainable=False),
			'w_6':tf.Variable(tf.random_normal([3,3,256,256]),trainable=False),
			'w_7':tf.Variable(tf.random_normal([3,3,256,256]),trainable=False)
		}

		self.b = {
			#1
			'w_1':tf.Variable(tf.zeros([64]),trainable=False),
			'w_2':tf.Variable(tf.zeros([64]),trainable=False),
			#2
			'w_3':tf.Variable(tf.zeros([128]),trainable=False),
			'w_4':tf.Variable(tf.zeros([128]),trainable=False),
			#3
			'w_5':tf.Variable(tf.zeros([256]),trainable=False),
			'w_6':tf.Variable(tf.zeros([256]),trainable=False),
			'w_7':tf.Variable(tf.zeros([256]),trainable=False)
		}

		weights = np.load(weight_file)
		keys = sorted(weights.keys())
		count = 1
		for i,k in enumerate(keys):
			if i<7:
				index = 'w_' + str(count)
				print(i,k,np.shape(weights[k]))
				if i%2==0:
					sess.run(self.w[index].assign(weights[k]))
				else:
					sess.run(self.b[index].assign(weights[k]))
					count += 1	
			else:
				break

	
	def vgg_16(self,x):
		layer = tf.reshape(x,shape=[-1,self.shape,self.shape,3])
		layer = tf.subtract(layer,vgg_mean_bgr)
		max_pool = [0,1,0,1,0,0,1]

		for i in range(0,7):
			index = 'w_'+str(i+1)		
			layer = conv_layer(layer,self.w[index],self.b[index],max_pool=max_pool[i])
			if i==self.layer_no:
				vgg_layer = layer
				break

		return vgg_layer


	
		