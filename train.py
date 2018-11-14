import tensorflow as tf
from gan import gan
from helper_func import load_batch,test_network
#from opti import constrained_problem
from tqdm import tqdm
import os,glob,cv2
from vgg16 import vgg
import numpy as np

batch_size = 25
steps = int(25000/batch_size)
shape = 32
scaling_res = 2

train = tf.placeholder(tf.bool)
x = tf.placeholder('float',[None,shape,shape,3])
y = tf.placeholder('float',[None,shape*scaling_res,shape*scaling_res,3])
lr = tf.placeholder('float')

def train_network(x,y,train,lr):
	gan_model = gan(shape,scaling_res)
	generator = gan_model.gen_model(x,train)
	fake_prediction = gan_model.dis_model(generator,train)
	real_prediction = gan_model.dis_model(y,train)

	with tf.Session() as sess:
		try:
			vgg_model = vgg(sess,shape*scaling_res)
			fake_features = vgg_model.vgg_16(generator)
			real_features = vgg_model.vgg_16(y)
			dis_cost = tf.losses.mean_squared_error(tf.ones_like(real_prediction),real_prediction) + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.zeros_like(fake_prediction),logits=fake_prediction))
			gen_cost = tf.losses.mean_squared_error(real_features,fake_features) + 0.01*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.ones_like(fake_prediction),logits=fake_prediction))
			gen_optimizer = tf.train.AdamOptimizer(learning_rate=lr*2.5)
			dis_optimizer = tf.train.RMSPropOptimizer(learning_rate=lr*0.01)
			first_summary = tf.summary.scalar(name='gen_cost',tensor=gen_cost)
			second_summary = tf.summary.scalar(name='dis_cost',tensor=dis_cost)
			writer = tf.summary.FileWriter('./graphs',sess.graph)
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				gen_train_op = gen_optimizer.minimize(gen_cost)
				dis_train_op = dis_optimizer.minimize(dis_cost)
			saver = tf.train.Saver(tf.global_variables())
			

			sess.run(tf.global_variables_initializer())
			count = 0
			patience_lr = 0
			w_damp = 1.0
			x_train,y_train = load_batch('/home/sidhanth/project_workspace/init_dataset/d2/*.png',shape)
			#x_test,y_test = load_batch(test_path,shape)
			#prev_gen_loss = float("inf")

			for epoch in range(100):
				gen_epoch_loss = 0.0
				dis_epoch_loss = 0.0
				#gen_val_loss = 0.0
				#dis_val_loss = 0.0
				
				with tqdm(total=steps) as t:	
					for step in range(steps):
						feed = {x:x_train[step*batch_size:(step+1)*batch_size], y:y_train[step*batch_size:(step+1)*batch_size], train:True, lr:0.1*w_damp}
						
						for i in range(3):
							_,dis_loss = sess.run([dis_train_op,dis_cost],feed_dict=feed)
						_,gen_loss = sess.run([gen_train_op,gen_cost],feed_dict=feed)
						
						dis_epoch_loss += dis_loss
						gen_epoch_loss += gen_loss
						summary_1 = sess.run(first_summary,feed_dict=feed)
						summary_2 = sess.run(second_summary,feed_dict=feed)
						writer.add_summary(summary_1,count)
						writer.add_summary(summary_2,count)
						count+=1
						t.set_postfix_str('Epoch:'+str(epoch+1)+' (gen_loss,dis_loss):('+str(gen_epoch_loss/(step+1))+', '+str(dis_epoch_loss/(step+1))+')')
						t.update()
				print('Epoch Completed:',epoch+1)

				"""
				step_size = int(800/batch_size)
				for step in range(step_size):
					feed = {x:x_test[step*batch_size:(step+1)*batch_size], y:y_test[step*batch_size:(step+1)*batch_size], train:False, lr:0.1*w_damp}
					gen_loss,dis_loss = sess.run([gen_cost,dis_cost],feed_dict=feed)
					gen_val_loss += gen_loss
					dis_val_loss += dis_loss
				print('Validation losses: '+str(gen_val_loss/step_size)+str(dis_val_loss/2*step_size))

				if(prev_gen_loss>gen_val_loss)
					print('Saving best model. Percentage change in test loss:',100*(prev_gen_loss-gen_val_loss)/prev_gen_loss)
					saver.save(sess,'./model.ckpt')
					prev_gen_loss = gen_val_loss

				else:
					print('Test loss did not improve.')
					patience_lr++
					if patience_lr >= 10*(w_damp):
						print('Reducing learning rate on plateau')
						patience_lr = 0
						w_damp *= 0.998
				"""

		except KeyboardInterrupt:
			saver.save(sess,'./model.ckpt')
			print('Model Saved')

train_network(x,y,train,lr)


	

	
