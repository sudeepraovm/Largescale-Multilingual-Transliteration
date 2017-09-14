#!/usr/bin/python
# -*- coding:utf-8-*-

import os.path
import operator
import pickle
from math import sqrt
import numpy as np 


class Vocab:
	def __init__(self):
		self.word_to_index = {}
		self.index_to_word = {}
		self.unknown       = "<unk>"
		self.end_of_sym    = "<eos>"
		self.start_sym     = "<s>"
		self.padding       = "<pad>"
		self.word_freq     = {}
		self.len_vocab     = 0
		self.total_words   = 0
		self.embeddings    = None
	
	def add_constant_tokens(self):
		self.word_to_index[self.padding] = 0 
		self.word_to_index[self.unknown] = 1
		self.word_to_index[self.end_of_sym] = 2
		self.word_to_index[self.start_sym] = 3
	
	def add_word(self, word):

		if word in self.word_to_index:
			self.word_freq[word] += 1

		else:
			self.word_to_index[word] = len(self.word_to_index)
			#print word
			#print self.word_to_index[word]
			self.word_freq[word] = 1	  	

	def create_reverse_dictionary(self):

		for key, val in self.word_to_index.iteritems():
			self.index_to_word[val] = key

	def construct_dictionary_single_file(self, filename):

		with open(filename, 'r') as f:
			f = f.read()
			for line in f.split('\n'):
				try:
					x = line.split(':')[1]
					enword = x.split('?')[0]
					lword  = x.split('?')[1][:-1]
					#print lword
					for i in unicode(lword, 'utf-8'):
						
						self.add_word(i)

				except IndexError:
					print 'IndexError'
	
	def construct_dictionary_multiple_files(self, filenames):

		for files in filenames:
			self.construct_dictionary_single_file(files)

	def encode_word(self, word):
		if word not in self.word_to_index:
			word = self.unknown

		return self.word_to_index[word]

	def decode_word(self, index):

		if index not in self.index_to_word:
			return self.unknown

		return self.index_to_word[index]

	def get_embeddings(self, embedding_size):

		sorted_list = sorted(self.index_to_word.items(), key = operator.itemgetter(0))
		embeddings = []

		np.random.seed(1357)

		for index, word in sorted_list:

			if word in ['<pad>', '<s>', '<eos>']:
				temp = np.zeros((embedding_size))
			else:
				temp = np.random.uniform(-sqrt(3)/sqrt(embedding_size), sqrt(3)/sqrt(embedding_size), (embedding_size))

			embeddings.append(temp)

		self.embeddings = np.asarray(embeddings)
		self.embeddings = self.embeddings.astype(np.float32)

	def construct_vocab(self, filenames, embedding_size):

		self.construct_dictionary_multiple_files(filenames)
		self.add_constant_tokens()
		self.create_reverse_dictionary()
		self.get_embeddings(embedding_size)

		self.len_vocab = len(self.word_to_index)

		print ("Number of words in the vocabulary is " + str(len(self.word_to_index)))

		self.total_words = float(sum(self.word_freq.values()))
	

def main():
	vocab = Vocab()
	filenames = ['new/en_data.txt', 'new/hi_data.txt']
	vocab.construct_vocab(filenames, 20)
	for i in vocab.index_to_word:
		print vocab.index_to_word[i].encode('utf-8') 
	print vocab.index_to_word[194]
	print vocab.encode_word('프'.decode('utf-8'))


if __name__ == '__main__':
	main()