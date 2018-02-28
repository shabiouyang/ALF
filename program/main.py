#Author: Wenzhe Ouyang
#Strating time: 2017/12/23

import numpy as np
import tensorflow as tf
import load_image
import cv2

def xavier_init(fan_in, fan_out, constant = 1):
	low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
	high = constant * np.sqrt(6.0 / (fan_in + fan_out))
	return tf.random_uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)

class autoEncoder(object):
	def __init__(self, num_inputs, num_shots, activation_function = None, 
		optimizer = tf.train.AdamOptimizer()):
		self.num_inputs = num_inputs
		self.num_shots = num_shots
		self.AF = activation_function
		networks_weights = self._initialize_weights()
		self.weights = networks_weights

		self.x = tf.placeholder(tf.float32, [None, self.num_inputs])
		
		if self.AF == None:
			self.num_shots = tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1'])
		else:
			self.num_shots = self.AF(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
		
		self.reconstruction = tf.add(tf.matmul(self.num_shots, self.weights['w2']), self.weights['b2'])
		
		self.loss = tf.losses.mean_squared_error(labels=self.x, predictions=self.reconstruction)
		self.optimizer = optimizer.minimize(self.loss)
		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

	def _initialize_weights(self):
		all_weights = {}
		all_weights['w1'] = tf.Variable(xavier_init(self.num_inputs, self.num_shots))
		all_weights['b1'] = tf.Variable(tf.zeros([self.num_shots], dtype=tf.float32))
		all_weights['w2'] = tf.Variable(tf.zeros([self.num_shots, self.num_inputs], dtype=tf.float32))
		all_weights['b2'] = tf.Variable(tf.zeros([self.num_inputs], dtype=tf.float32))
		return all_weights

	def partial_fit(self, X):
		loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict = {self.x: X})
		return loss
	def transform(self, X):
		return self.sess.run(self.hidden, feed_dict = {self.x: X})
	def generate(self, encoded = None):
		if hidden is None:
			hidden = np.random.normal(size = self.weights[b1])
		return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})
	def reconstruct(self, X):
		return self.sess.run(self.reconstruction, feed_dict = {self.x: X})
	def get_weights(self):
		return self.sess.run(self.weights['w1'])
	def get_biases(self):
		return self.sess.run(self.weights['b1'])

def get_random_block_from_data(data, batch_size):
	start_index = np.random.randint(0, len(data) - batch_size)
	return data[start_index:(start_index + batch_size)]

def get_train_data():
	data1, width1, height1 = load_image.loadimage("Knights")
	data2, width2, height2 = load_image.loadimage("Bunny")
	data3, width3, height3 = load_image.loadimage("Amethyst")
	data4, width4, height4 = load_image.loadimage("Jelly Beans")
	data5, width5, height5 = load_image.loadimage("dice")

	data_1 = np.r_[data1[:,:,0],data1[:,:,1],data1[:,:,2]]
	data_2 = np.r_[data2[:,:,0],data2[:,:,1],data2[:,:,2]]
	data_3 = np.r_[data3[:,:,0],data3[:,:,1],data3[:,:,2]]
	data_4 = np.r_[data4[:,:,0],data4[:,:,1],data4[:,:,2]]
	data_5 = np.r_[data4[:,:,0],data4[:,:,1],data4[:,:,2]]


	train_data = np.r_[data_1, data_2, data_3, data_4, data_5]
	train_length = len(train_data)
	return train_data, train_length

def get_test_data():
	test, width, height = load_image.loadimage("DragonsAndBunnies")
	test_data = np.r_[test[:,:,0], test[:,:,1], test[:,:,2]]
	test_length = len(test_data)
	return test_data, test_length, width, height

def main():
	train_data, train_length = get_train_data()
	test_data, test_length, width, height = get_test_data()

	#saver = tf.train.Saver()
	training_epoch = 20
	batch_size = 64
	autoencoder = autoEncoder(num_inputs=25, num_shots = 5, activation_function = None, 
		optimizer=tf.train.AdamOptimizer(learning_rate = 0.001))
	for epoch in range(training_epoch):
		print("training times: ", epoch)
		total_batch = int(train_length/batch_size)

		for i in range(total_batch):
			batch_xs = get_random_block_from_data(train_data, batch_size)
			loss = autoencoder.partial_fit(batch_xs)
			#This is for saving network
			#saver.save(sess,'./my_net/autoencoder.ckpt', global_step = epoch+1)
	W = autoencoder.get_weights()
	b = autoencoder.get_biases()
	print(W)
	print(b)
	output_images = autoencoder.reconstruct(test_data)
	output_length= len(output_images)
	rgb_length = int(output_length/3)
	reconstruct_subimages = np.zeros([height*width, 25, 3], dtype=np.float32)
	for channels in range(3):
		reconstruct_subimages[:, :, int(channels)] = output_images[(rgb_length*channels):(rgb_length*(channels+1)),:]
	all_image_rec = np.zeros([25,height,width,3], dtype=np.float32)
	for x in range(width):
		for y in range(height):
			all_image_rec[:,y,x,:] = reconstruct_subimages[y * width + x,:]*255
	for numbers in range(25):
		all_image_rec[numbers, :, :, :] = cv2.cvtColor(all_image_rec[numbers, :, :, :], cv2.COLOR_BGR2RGB)
		cv2.imwrite("./Reconstruct/" + str(numbers+1) + "_" + "autoencoder" + ".png", all_image_rec[numbers, :, :, :])
if __name__ == '__main__':
	main()