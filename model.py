#!/usr/bin/python
# -*- coding:utf-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import math
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from seq2seq import *
from dataiterator import *
import os
import numpy as np


class Config:

	def __init__(self, learning_rate = .0001, embedding_size = 512, hidden_size = 512, batch_size = 128, max_epoch = 50,
				max_sequence_length_lang1 = 20, max_sequence_length_lang2 =20):

		config_file = open('config.txt', 'w')

		self.learning_rate = learning_rate
		self. embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.max_epoch = max_epoch
		self.max_sequence_length_lang1 = max_sequence_length_lang1
		self.max_sequence_length_lang2 = max_sequence_length_lang2
		

		config_file.write("Learning rate " + str(self.learning_rate) + "\n")
		config_file.write("Embedding size " + str(self.embedding_size) + "\n")
		config_file.write("hidden size " + str(self.hidden_size) + "\n")
		config_file.write("Batch size " + str(self.batch_size) + "\n")
		config_file.write("Max Epochs" + str(self.max_epoch) + "\n")
		config_file.close() 


class run_model:
	
	def __init__(self, lang1_file, bA, lang2_file, config = None):

		if config is None:
			config = Config()

		self.config = config
		self.model  = bA

		self.dataset = PadDataset(lang1_file, lang2_file, self.config.embedding_size)


	def add_placeholders(self):

		self.encoder_input_placeholder = tf.placeholder(tf.int32, shape= (self.config.max_sequence_length_lang1, None), name = 'encode')
		self.decoder_input_placeholder = tf.placeholder(tf.int32, shape= (self.config.max_sequence_length_lang2, None), name = 'decoder')
		self.label_placeholder         = tf.placeholder(tf.int32, shape= (self.config.max_sequence_length_lang2, None), name = 'label')
		self.weights_placeholder       = tf.placeholder(tf.int32, shape= (self.config.max_sequence_length_lang2, None), name = 'weights')
		self.feed_previous_placeholder = tf.placeholder(tf.bool, name = 'feed_previous')
		
	def fill_feed_dict(self, encoder_inputs, decoder_inputs, labels, weights, feed_previous = False):

		feed_dict = {
		self.encoder_input_placeholder : encoder_inputs,
		self.decoder_input_placeholder : decoder_inputs,
		self.label_placeholder         : labels,
		self.weights_placeholder       : weights,
		self.feed_previous_placeholder : feed_previous, 
		}

		return feed_dict

	def run_epoch(self, epoch_no, sess, fp = None):

		start_time = time.time()
		#print self.dataset.datasets[0].lang1
		steps_per_epoch = int(math.ceil(float(self.dataset.datasets[0].num_samples)) / float(self.config.batch_size))
		total_loss = 0
		for step in xrange(steps_per_epoch):
			lang1_data, lang2_data, title_data, weights, max_lang1, max_lang2 = self.dataset.next_batch(
				self.dataset.datasets[0], self.config.batch_size, True)
			if fp is None:
				if (epoch_no > 5):
					feed_previous = True
				else:
					feed_previous = False
			else:
				feed_previous = fp	

			feed_dict = self.fill_feed_dict(lang1_data, title_data, lang2_data, weights, feed_previous = True)

			_, loss_value, outputs	= sess.run([self.train_op, self.loss_ops, self.logits], feed_dict = feed_dict)
			total_loss += loss_value

			duration = time.time() - start_time
			print ('loss_value', loss_value, ' ', step)
			sys.stdout.flush()

			if (step + 1 == steps_per_epoch) or ((step + 1) % 5000 == 0):

				print('Step %d: Loss = %.2f'% (step, loss_value))
				sys.stdout.flush()
				
				print('Training Data Eval:')
				self.print_titles(sess, self.dataset.datasets[0], 7)
					
				
				print('Step %d: loss = %.2f' % (step, loss_value))
				print('Validation Data Eval:')
				loss_value = self.do_eval(sess,self.dataset.datasets[2])
				self.print_titles(sess,self.dataset.datasets[2], 2)
			   
				print('Test Data Eval:')
				loss_value = self.do_eval(sess,self.dataset.datasets[1])
				self.print_titles(sess,self.dataset.datasets[1], 2)
				print('Step %d: loss = %.2f' % (step, loss_value))

				self.print_titles_in_files(sess, self.dataset.datasets[0])
				self.print_titles_in_files(sess, self.dataset.datasets[1])
				self.print_titles_in_files(sess, self.dataset.datasets[2])
				sys.stdout.flush()

		return float(total_loss) / float(steps_per_epoch)        


	def do_eval(self, sess, data_set):

		total_loss = 0
		steps_per_epoch = int(math.ceil(float(data_set.num_samples)) / float(self.config.batch_size))

		for step in xrange(steps_per_epoch):
			lang1_data, lang2_data, title_data, weights, max_lang1, max_lang2 = self.dataset.next_batch(
				data_set, self.config.batch_size, True)

			feed_dict = self.fill_feed_dict(lang1_data, title_data, lang2_data, weights, feed_previous = True)
			loss_value = sess.run(self.loss_ops, feed_dict = feed_dict)
			total_loss += loss_value

		return float(total_loss)/float(steps_per_epoch)

	def run_training(self):

		
		with tf.Graph().as_default():

			conf = tf.ConfigProto(device_count = {'GPU': 0})

			tf.set_random_seed(1357)
			
			self.config.max_sequence_length_lang1 = max(val.max_lang1 for i, val in self.dataset.datasets.iteritems())
			self.config.max_sequence_length_lang2 = max(val.max_lang2 for i, val in self.dataset.datasets.iteritems())

			len_vocab = self.dataset.length_vocab()
			self.add_placeholders()

			self.logits = self.model.inference(self.encoder_input_placeholder, self.decoder_input_placeholder, self.config.embedding_size,
												self.feed_previous_placeholder, len_vocab, self.config.hidden_size, 
												self.weights_placeholder)

			self.loss_ops = self.model.loss_ops( self.logits, self.label_placeholder, self.weights_placeholder, len_vocab)

			self.train_op = self.model.training( self.loss_ops, self.config.learning_rate)

			init = tf.global_variables_initializer()

			saver = tf.train.Saver()
			sess = tf.Session(config = conf)

			summary_writer = tf.summary.FileWriter('logs', sess.graph)

			if (os.path.exists('last_model')):
				saver.restore(sess, last_model)

			else:
				sess.run(init)
			best_val_loss = float('inf')
			best_val_epoch = 0	

			for epoch in xrange(self.config.max_epoch):
				print ('Epoch: '+ str(epoch))
				start = time.time()

				train_loss = self.run_epoch(epoch, sess)
				valid_loss = self.do_eval(sess, self.dataset.datasets[2])

				print ('training loss:{}'.format(train_loss))
				print ('Validation loss:{}'.format(valid_loss))

				if (valid_loss<= best_val_loss):
					best_val_loss = valid_loss
					best_val_epoch = epoch 
					saver.save(sess, 'best_model')

				if (epoch == self.config.max_epoch-1):
					saver.save(sess, 'last_model')

				print ("Total time:{}".format(time.time() - start))

			saver.restore(sess, 'best_model')
			test_loss = self.do_eval(sess, self.dataset.datasets[1])
			print ("Test Loss:{}".format(test_loss))	
			self.print_titles_in_files(sess, self.dataset.datasets[1])
			self.print_titles_in_files(sess, self.dataset.datasets[2])


	def print_titles(self, sess, data_set, total_examples):

		lang1_data, lang2_data, title_data, weights, max_lang1, max_lang2 = self.dataset.next_batch(
				data_set, self.config.batch_size, False)

		feed_dict = self.fill_feed_dict(lang1_data, title_data, lang2_data, weights, feed_previous = True)

		_decoder_states = sess.run(self.logits, feed_dict = feed_dict)
		decoder_states = np.array([np.argmax(i, axis=1) for i in _decoder_states])

		ds = np.transpose(decoder_states)
		true_labels = np.transpose(lang2_data)

		final_ds = ds.tolist()
		true_labels = true_labels.tolist() 
		for i, state in enumerate(final_ds):
			print ('Predicted translitration is '+ self.dataset.decode_to_sentence(state))
			print ('True translitration is' + self.dataset.decode_to_sentence(true_labels[i]))


	def print_titles_in_files(self, sess, data_set):


		f1 = open(data_set.name +'_final_result', 'wb')
		lang1_data, lang2_data, title_data, weights, max_lang1, max_lang2 = self.dataset.next_batch(
				data_set, self.config.batch_size, False)

		feed_dict = self.fill_feed_dict(lang1_data, title_data, lang2_data, weights, feed_previous = True)

		_decoder_states = sess.run(self.logits, feed_dict = feed_dict)
		decoder_states = np.array([np.argmax(i, axis = 0) for i in _decoder_states])

		ds = decoder_states
		true_labels = lang2_data

		final_ds = ds.tolist()
		true_labels = true_labels.tolist()

		for i, state in enumerate(final_ds):
			s =  self.dataset.decode_to_sentence(state)
			t =  self.dataset.decode_to_sentence(true_labels[i])
			f1.write(s + "\n")
			f1.write(t +"\n")		









class Basic_model:


	def add_cell(self, hidden_size, cell_input = None):

		if cell_input is None:
			self.enc_cell = tf.nn.rnn_cell.GRUCell(hidden_size)

		else:
			self.enc_cell = cell_input
	
	def add_projection_layer(self, hidden_size, len_vocab):
		
		self.projection_B = tf.get_variable(name="Projection_B", shape=[len_vocab])
		self.projection_W = tf.get_variable(name="Projected_W", shape=[hidden_size, len_vocab])		

	def inference(self, encoder_inputs, decoder_inputs, embedding_size, feed_previous, len_vocab, hidden_size, 
					weights, c = None ):

		self.add_cell(hidden_size, c)
		self.add_projection_layer(hidden_size, len_vocab)

		ei = tf.unstack(encoder_inputs)
		di = tf.unstack(decoder_inputs)

		outputs, states = embedding_attention_seq2seq(encoder_inputs = ei,
								decoder_inputs = di,
								cell = self.enc_cell,
								num_encoder_symbols = len_vocab,
								num_decoder_symbols = len_vocab,
								embedding_size = embedding_size,
								num_heads=1,
								output_projection = (self.projection_W, self.projection_B),
								feed_previous = feed_previous,
								dtype = tf.float32,
								scope = None,
								initial_state_attention=False)

		self.final_output = [tf.matmul(o, self.projection_W) + self.projection_B  for o in outputs]

		return self.final_output


	def loss_ops(self, outputs, labels, weights, len_vocab):	

		_labels = tf.unstack(labels)
		weights = tf.to_float(weights)
		_weights = tf.unstack(weights)

		loss_per_batch = tf.contrib.legacy_seq2seq.sequence_loss(outputs, _labels, _weights)

		self.loss_per_batch = loss_per_batch
		return loss_per_batch

	def training(self, loss, learning_rate):
		
		optimizer = tf.train.AdamOptimizer(learning_rate)
		train_op = optimizer.minimize(loss)
		return train_op

	

def main():
	
	runModel = run_model('new/en_data.txt', Basic_model(), 'new/hi_data.txt')
	runModel.run_training()


if __name__ == '__main__':
	main()			

					
		

		



		
