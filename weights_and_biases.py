import tensorflow as tf
from utils import variable

def params(trainable_1,trainable_2,trainable_3):
      weights = {
                  '1_w_conv1':variable(1,1,[3,3,3,32],trainable_1),
                  '1_w_conv2':variable(1,2,[3,3,32,128],trainable_1),
                  '1_w_conv3':variable(1,3,[3,3,128,128],trainable_1),
                  '1_w_conv4':variable(1,4,[3,3,128,128],trainable_1),
                  '1_w_conv5':variable(1,5,[5,5,128,256],trainable_1),
                  '1_out':variable(1,6,[1,1,64,3],trainable_1),

                  '2_w_conv1':variable(2,1,[3,3,3,32],trainable_2),
                  '2_w_conv2':variable(2,2,[3,3,32,128],trainable_2),
                  '2_w_conv3':variable(2,3,[3,3,128,128],trainable_2),
                  '2_w_conv4':variable(2,4,[3,3,128,128],trainable_2),
                  '2_w_conv5':variable(2,5,[5,5,128,128],trainable_2),
                  '2_w_conv6':variable(2,6,[5,5,128,128],trainable_2),
                  '2_w_conv7':variable(2,7,[7,7,128,256],trainable_2),
                  '2_out':variable(2,8,[1,1,64,3],trainable_2),
                  '2_w_conv9':variable(2,9,[3,3,6,128],trainable_2),
                  '2_w_conv10':variable(2,10,[3,3,128,256],trainable_2),
                  '4x_out':variable(2,11,[1,1,64,3],trainable_2),

                  '3_w_conv1':variable(3,1,[3,3,3,32],trainable_3),
                  '3_w_conv2':variable(3,2,[3,3,32,128],trainable_3),
                  '3_w_conv3':variable(3,3,[3,3,128,128],trainable_3),
                  '3_w_conv4':variable(3,4,[3,3,128,128],trainable_3),
                  '3_w_conv5':variable(3,5,[5,5,128,128],trainable_3),
                  '3_w_conv6':variable(3,6,[5,5,128,128],trainable_3),
                  '3_w_conv7':variable(3,7,[5,5,128,128],trainable_3),
                  '3_w_conv8':variable(3,8,[7,7,128,128],trainable_3),
                  '3_w_conv9':variable(3,9,[7,7,128,256],trainable_3),
                  '3_out':variable(3,10,[1,1,64,3],trainable_3),
                  '3_w_conv11':variable(3,11,[3,3,6,128],trainable_3),
                  '3_w_conv12':variable(3,12,[3,3,128,256],trainable_3),
                  '3_w_conv13':variable(3,13,[3,3,64,3],trainable_3),
                  '3_w_conv14':variable(3,14,[5,5,6,128],trainable_3),
                  '3_w_conv15':variable(3,15,[7,7,128,256],trainable_3),
                  '8x_out':variable(3,16,[1,1,64,3],trainable_3),

                  'dis_w_conv1':variable(4,1,[3,3,3,32],True),
                  'dis_w_conv2':variable(4,2,[3,3,32,64],True),
                  'dis_w_conv3':variable(4,3,[3,3,64,128],True),
                  'dis_w_conv4':variable(4,4,[3,3,128,256],True),
                  'dis_w_fc':variable(4,5,[8*8*256,1024],True),
                  'dis_out':variable(4,6,[1024,16],True)
      }

      biases = {
                  '1_b_conv1':tf.Variable(tf.zeros([32]),trainable=trainable_1),
                  '1_b_conv2':tf.Variable(tf.zeros([128]),trainable=trainable_1),
                  '1_b_conv3':tf.Variable(tf.zeros([128]),trainable=trainable_1),
                  '1_b_conv4':tf.Variable(tf.zeros([128]),trainable=trainable_1),
                  '1_b_conv5':tf.Variable(tf.zeros([256]),trainable=trainable_1),
                  '1_out':tf.Variable(tf.zeros([3]),trainable=trainable_1),

                  '2_b_conv1':tf.Variable(tf.zeros([32]),trainable=trainable_2),
                  '2_b_conv2':tf.Variable(tf.zeros([128]),trainable=trainable_2),
                  '2_b_conv3':tf.Variable(tf.zeros([128]),trainable=trainable_2),
                  '2_b_conv4':tf.Variable(tf.zeros([128]),trainable=trainable_2),
                  '2_b_conv5':tf.Variable(tf.zeros([128]),trainable=trainable_2),
                  '2_b_conv6':tf.Variable(tf.zeros([128]),trainable=trainable_2),
                  '2_b_conv7':tf.Variable(tf.zeros([256]),trainable=trainable_2),
                  '2_out':tf.Variable(tf.zeros([3]),trainable=trainable_2),
                  '2_b_conv9':tf.Variable(tf.zeros([128]),trainable=trainable_2),
                  '2_b_conv10':tf.Variable(tf.zeros([256]),trainable=trainable_2),
                  '4x_out':tf.Variable(tf.zeros([3]),trainable=trainable_2),

                  '3_b_conv1':tf.Variable(tf.zeros([32]),trainable=trainable_3),
                  '3_b_conv2':tf.Variable(tf.zeros([128]),trainable=trainable_3),
                  '3_b_conv3':tf.Variable(tf.zeros([128]),trainable=trainable_3),
                  '3_b_conv4':tf.Variable(tf.zeros([128]),trainable=trainable_3),
                  '3_b_conv5':tf.Variable(tf.zeros([128]),trainable=trainable_3),
                  '3_b_conv6':tf.Variable(tf.zeros([128]),trainable=trainable_3),
                  '3_b_conv7':tf.Variable(tf.zeros([128]),trainable=trainable_3),
                  '3_b_conv8':tf.Variable(tf.zeros([128]),trainable=trainable_3),
                  '3_b_conv9':tf.Variable(tf.zeros([256]),trainable=trainable_3),
                  '3_out':tf.Variable(tf.zeros([3]),trainable=trainable_3),
                  '3_b_conv11':tf.Variable(tf.zeros([128]),trainable=trainable_3),
                  '3_b_conv12':tf.Variable(tf.zeros([256]),trainable=trainable_3),
                  '3_b_conv13':tf.Variable(tf.zeros([3]),trainable=trainable_3),
                  '3_b_conv14':tf.Variable(tf.zeros([128]),trainable=trainable_3),
                  '3_b_conv15':tf.Variable(tf.zeros(256),trainable=trainable_3),
                  '8x_out':tf.Variable(tf.zeros(3),trainable=trainable_3),

                  'dis_b_conv1':tf.Variable(tf.zeros([32])),
                  'dis_b_conv2':tf.Variable(tf.zeros([64])),
                  'dis_b_conv3':tf.Variable(tf.zeros([128])),
                  'dis_b_conv4':tf.Variable(tf.zeros([256])),
                  'dis_b_fc':tf.Variable(tf.zeros([1024])),
                  'dis_out':tf.Variable(tf.zeros([16]))
      }

      return weights,biases
