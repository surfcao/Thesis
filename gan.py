import tensorflow as tf
from weights_and_biases import params
from generator import generator
from discriminator import discriminator
from utils import conv2d,prelu,lrelu,batch_norm,tanh,subpixel2d


class gan:
	def __init__(self,shape,final_res):
		self.shape = shape
		self.final_res = final_res
		
		if self.final_res==2:
			self.weights, self.biases = params(True,False,False)
		
		elif self.final_res==4:
			self.weights, self.biases = params(False,True,False)
		
		elif self.final_res==8:
			self.weights, self.biases = params(False,False,True)

	def gen_model(self,x,train):
		if self.final_res%2==0:
			gen_1 = generator(self.weights,self.biases,1,self.shape)
			out_1 = gen_1.gen(x,train)
			out = out_1
			
		if self.final_res%4==0:
			gen_2 = generator(self.weights,self.biases,2,self.shape)
			out_2 = gen_2.gen(x,train)
			out_2 = tf.concat([out_2,out_1],3)
			
			out_2 = conv2d(out_2,self.weights['2_w_conv9']) + self.biases['2_b_conv9']
			out_2 = batch_norm(out_2,train,name='concat_1')
			out_2 = prelu(out_2,'concat_batchnorm1')
			
			out_2 = conv2d(out_2,self.weights['2_w_conv10']) + self.biases['2_b_conv10']
			out_2 = subpixel2d(out_2,[-1,shape*4,shape*4,64])
			out_2 = prelu(out_2,'concat_subpixel1')
			
			out_2 = conv2d(out_2,self.weights['4x_out']) + self.biases['4x_out'] 	
			out_2 = 127.5*(tanh(out_2)+1)
			out = out_2

		if self.final_res%8==0:
			gen_3 = generator(self.weights,self.biases,3,self.shape)
			out_3 = gen_3.gen(x,train)
			out_3 = tf.concat([out_3,out_1],3)
			
			out_3 = conv2d(out_3,self.weights['3_w_conv11']) + self.biases['3_b_conv11']
			out_3 = batch_norm(out_3,train,name='concat_2')
			out_3 = prelu(out_3,'concat_batchnorm2')

			out_3 = conv2d(out_3,self.weights['3_w_conv12']) + self.biases['3_b_conv12']
			out_3 = subpixel2d(out_3,[-1,shape*4,shape*4,64])
			out_3 = prelu(out_3,'concat_subpixel2')

			out_3 = conv2d(out_3,self.weights['3_w_conv13']) + self.biases['3_b_conv13']
			out_3 = lrelu(out_3)
			out_3 = tf.concat([out_3,out_2],3)

			out_3 = conv2d(out_3,self.weights['3_w_conv14']) + self.biases['3_b_conv14']
			out_3 = batch_norm(out_3,train,name='concat_3')
			out_3 = prelu(out_3,'concat_batchnorm3')

			out_3 = conv2d(out_3,self.weights['3_w_conv15']) + self.biases['3_b_conv15']
			out_3 = subpixel2d(out_3,[-1,shape*8,shape*8,64])
			out_3 = prelu(out_3,'concat_subpixel3')

			out_3 = conv2d(out_3,self.weights['8x_out']) + self.biases['8x_out']
			out_3 = 127.5*(tanh(out_3)+1)
			out = out_3
		
		return out

	def dis_model(self,x,train):
		dis = discriminator(self.weights,self.biases,self.shape*self.final_res)
		out = dis.dis(x,train)
		return out 
