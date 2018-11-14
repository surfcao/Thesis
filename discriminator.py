import tensorflow as tf
from utils import conv2d,maxpool2d,lrelu,prelu,tanh

class discriminator:
    def __init__(self,weights,biases,shape,keep_rate=0.8):
        self.weights = weights
        self.biases = biases
        self.shape = shape
        self.keep_rate = keep_rate

    def dis(self,x,training):
        x = tf.reshape(x,shape=[-1,self.shape,self.shape,3])
        scope = 'dis_'
        layer = lrelu(conv2d(x,self.weights[scope+'w_conv1'])+self.biases[scope+'b_conv1'])
        
        for i in range(1,4):
        	conv = prelu(conv2d(layer,self.weights[scope+'w_conv'+str(i+1)])+self.biases[scope+'b_conv'+str(i+1)],scope+'w_conv'+str(i+1))
        	conv = maxpool2d(conv)
        	conv = tf.nn.dropout(conv,self.keep_rate)
        	layer = conv

        fc = tf.reshape(layer,[-1, int(self.shape/8)*int(self.shape/8)*256])
        fc = lrelu(tf.matmul(fc,self.weights[scope+'w_fc'])+self.biases[scope+'b_fc'])
        fc = tf.nn.dropout(fc,self.keep_rate)
        
        output = tf.matmul(fc,self.weights[scope+'out'])+self.biases[scope+'out']
        output = (tanh(output)+1.0)*0.5

        return output

