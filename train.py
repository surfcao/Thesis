import tensorflow as tf
from weights_bias import weights,biases
from generator import generator
from utils import conv2d,prelu,lrelu,batch_norm,tanh,subpixel2d
from helper_func import load_batch,test_network
from opti import constrained_problem
from tqdm import tqdm
import os,glob,cv2
from vgg16 import vgg
import numpy as np

# Do not forget to add UPDATE_OPS to trainable parameters as batch_norm function changed
batch_size = 16
steps = int(25000/batch_size)
shape = 64
scaling_res = 2

train = tf.placeholder(tf.bool)
x = tf.placeholder('float',[None,shape,shape,3])
y = tf.placeholder('float',[None,shape*scaling_res,shape*scaling_res,3])


def model(x,train,final_res):
	if final_res%2==0:
		gen_1 = generator(weights,biases,1)
		out_1 = gen_1.gen(x,train)
		out = out_1
		
	elif final_res%4==0:
		gen_2 = generator(weights,biases,2)
		out_2 = gen_2.gen(x,train)
		out_2 = tf.concat([out_2,out_1],3)
		
		out_2 = conv2d(out_2,weights['2_w_conv8']) + biases['2_b_conv8']
		out_2 = batch_norm(out_2,train,name='concat_1')
		out_2 = prelu(out_2,'concat_batchnorm1')
		
		out_2 = conv2d(out_2,weights['2_w_conv9']) + biases['2_b_conv9']
		out_2 = subpixel2d(out_2,[-1,shape*4,shape*4,64])
		out_2 = prelu(out_2,'concat_subpixel1')
		
		out_2 = conv2d(out_2,weights['4x_out']) + biases['4x_out'] 	
		out_2 = 127.5*(tanh(out_2)+1)
		out = out_2

	elif final_res%8==0:
		gen_3 = generator(weights,biases,3)
		out_3 = gen_3.gen(x,train)
		out_3 = tf.concat([out_3,out_1],3)
		
		out_3 = conv2d(out_3,weights['3_w_conv10']) + biases['3_b_conv10']
		out_3 = batch_norm(out_3,train,name='concat_2')
		out_3 = prelu(out_3,'concat_batchnorm2')

		out_3 = conv2d(out_3,weights['3_w_conv11']) + biases['3_b_conv11']
		out_3 = subpixel2d(out_3,[-1,shape*4,shape*4,64])
		out_3 = prelu(out_3,'concat_subpixel2')

		out_3 = conv2d(out_3,weights['3_w_conv12']) + biases['3_b_conv12']
		out_3 = lrelu(out_3)
		out_3 = tf.concat([out_3,out_2],3)

		out_3 = conv2d(out_3,weights['3_w_conv13']) + biases['3_b_conv13']
		out_3 = batch_norm(out_3,train,name='concat_3')
		out_3 = prelu(out_3,'concat_batchnorm3')

		out_3 = conv2d(out_3,weights['3_w_conv14']) + biases['3_b_conv14']
		out_3 = subpixel2d(out_3,[-1,shape*8,shape*8,64])
		out_3 = prelu(out_3,'concat_subpixel3')

		out_3 = conv2d(out_3,weights['8x_out']) + biases['8x_out']
		out_3 = 127.5*(tanh(out_3)+1)
		out = out_3
	
	return out
	

	
