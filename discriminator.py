import tensorflow as tf
from utils import conv2d,maxpool2d,lrelu,tanh

class discriminator:
    def __init__(self,weights,biases,shape,keep_rate=0.8):
        self.weights = weights
        self.biases = biases
        self.shape = shape
        self.keep_rate = keep_rate

    def dis(self,x,training):
        x = tf.reshape(x,shape=[-1,shape,shape,3])
        scope = 'dis_'
        layer_1 = lrelu(conv2d(x,weights[scope+'w_conv1'])+biases[scope+'b_conv1'])
        
        for i in range(1,4):
        	conv = prelu(conv2d(x,weights[scope+'w_conv'+str(i+1)])+biases[scope+'b_conv'+str(i+1)],scope+'w_conv'+str(i+1))
        	conv = maxpool2d(conv)
        	conv = tf.nn.dropout(conv,self.keep_rate)
        	x = conv

        fc = tf.reshape(x,[-1, (self.shape/16)*(self.shape/16)*256])
        fc = lrelu(tf.matmul(fc,weights[scope+'w_fc'])+biases[scope+'b_fc'])
        fc = tf.nn.dropout(fc,self.keep_rate)
        
        output = tf.matmul(fc,weights[scope+'out'])+biases[scope+'out']
        output = (tanh(output)+1.0)*0.5

        return output

