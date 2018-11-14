import tensorflow as tf
import numpy as np
import cv2,glob

def load_batch(path,init_shape,scale=2,end=25000):		
	image_path = glob.glob(path)
	count=1
	image_list_x = []
	image_list_y = []
	for image in image_path:
		if count >= end:
			break
		try:
			img = cv2.imread(image,1)
			img_x = cv2.resize(img,(init_shape,init_shape),interpolation=cv2.INTER_CUBIC)
			img_y = cv2.resize(img,(init_shape*scale,init_shape*scale),interpolation=cv2.INTER_CUBIC)
			#img_x = cv2.cvtColor(img_x,cv2.COLOR_BGR2RGB)
			#img_y = cv2.cvtColor(img_y,cv2.COLOR_BGR2RGB)
			image_list_x.append(img_x)
			image_list_y.append(img_y)
			print(count)
			count+=1
		except Exception as e:
			print(e)

	return np.array(image_list_x).astype('uint8'),np.array(image_list_y).astype('uint8')

	

def test_network(img_path,prediction,x,y,train,init_shape=128,scale=2):
	img = cv2.imread(img_path,1)	
	img_x,img_y = cv2.resize(img,(init_shape,init_shape),interpolation=cv2.INTER_CUBIC),cv2.resize(img,(init_shape*scale,init_shape*scale),interpolation=cv2.INTER_CUBIC)
	img_x,img_y = np.expand_dims(img_x,axis=0),np.expand_dims(img_y,axis=0)
	feed = {train:False,x:img_x,y:img_y}	
	img = prediction.eval(feed_dict=feed).astype('uint8')
	img = np.reshape(img,(init_shape*scale,init_shape*scale,3))
	image = y.eval(feed_dict=feed).astype('uint8')
	image = np.reshape(image,(init_shape*scale,init_shape*scale,3))
	cv2.imshow('true_image',image)
	cv2.imshow('generated_image',img)
	cv2.waitKey(0)

