import tensorflow as tf
from utils import variable

weights = {
			'1_w_conv1':variable(1,1,[3,3,3,32]),
      '1_w_conv2':variable(1,2,[3,3,32,128]),
      '1_w_conv3':variable(1,3,[3,3,128,128]),
      '1_w_conv4':variable(1,4,[3,3,128,128]),
      '1_w_conv5':variable(1,5,[5,5,128,256]),
      '1_out':variable(1,6,[1,1,64,3]),
      
      '2_w_conv1':variable(2,1,[3,3,3,32]),
      '2_w_conv2':variable(2,2,[3,3,32,128]),
      '2_w_conv3':variable(2,3,[3,3,128,128]),
      '2_w_conv4':variable(2,4,[3,3,128,128]),
      '2_w_conv5':variable(2,5,[5,5,128,128]),
      '2_w_conv6':variable(2,6,[5,5,128,128]),
      '2_w_conv7':variable(2,7,[7,7,128,256]),
      '2_out':variable(2,8,[1,1,64,3]),
      
      '3_w_conv1':variable(3,1,[3,3,3,32]),
      '3_w_conv2':variable(3,2,[3,3,32,128]),
      '3_w_conv3':variable(3,3,[3,3,128,128]),
      '3_w_conv4':variable(3,4,[3,3,128,128]),
      '3_w_conv5':variable(3,5,[5,5,128,128]),
      '3_w_conv6':variable(3,6,[5,5,128,128]),
      '3_w_conv7':variable(3,7,[5,5,128,128]),
      '3_w_conv8':variable(3,8,[7,7,128,128]),
      '3_w_conv9':variable(3,9,[7,7,128,256]),
      '3_out':variable(3,10,[1,1,64,3]),

      'dis_w_conv1':variable(4,1,[3,3,3,32]),
      'dis_w_conv2':variable(4,2,[3,3,32,64]),
      'dis_w_conv3':variable(4,3,[3,3,64,128]),
      'dis_w_conv4':variable(4,4,[3,3,128,256]),
      'dis_w_fc':variable(4,5,[16*16*256,1024]),
      'dis_out':variable(4,6,[1024,2])
}

biases = {
		   '1_b_conv1':tf.Variable(tf.zeros([32])),
       '1_b_conv2':tf.Variable(tf.zeros([128])),
       '1_b_conv3':tf.Variable(tf.zeros([128])),
       '1_b_conv4':tf.Variable(tf.zeros([128])),
       '1_b_conv5':tf.Variable(tf.zeros([256])),
       '1_out':tf.Variable(tf.zeros([3])),
       
       '2_b_conv1':tf.Variable(tf.zeros([32])),
       '2_b_conv2':tf.Variable(tf.zeros([128])),
       '2_b_conv3':tf.Variable(tf.zeros([128])),
       '2_b_conv4':tf.Variable(tf.zeros([128])),
       '2_b_conv5':tf.Variable(tf.zeros([128])),
       '2_b_conv6':tf.Variable(tf.zeros([128])),
       '2_b_conv7':tf.Variable(tf.zeros([256])),
       '2_out':tf.Variable(tf.zeros([3])),
       
       '3_b_conv1':tf.Variable(tf.zeros([32])),
       '3_b_conv2':tf.Variable(tf.zeros([128])),
       '3_b_conv3':tf.Variable(tf.zeros([128])),
       '3_b_conv4':tf.Variable(tf.zeros([128])),
       '3_b_conv5':tf.Variable(tf.zeros([128])),
       '3_b_conv6':tf.Variable(tf.zeros([128])),
       '3_b_conv7':tf.Variable(tf.zeros([128])),
       '3_b_conv8':tf.Variable(tf.zeros([128])),
       '3_b_conv9':tf.Variable(tf.zeros([256])),
       '3_out':tf.Variable(tf.zeros([3])),

       'dis_b_conv1':tf.Variable(tf.zeros([32])),
       'dis_b_conv2':tf.Variable(tf.zeros([64])),
       'dis_b_conv3':tf.Variable(tf.zeros([128])),
       'dis_b_conv4':tf.Variable(tf.zeros([256])),
       'dis_b_fc':tf.Variable(tf.zeros([1024])),
       'dis_out':tf.Variable(tf.zeros([2]))
}
