import tensorflow as tf
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



	

	
