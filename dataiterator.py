#!/usr/bin/python
# -*- coding:utf-8-*-

import random
import numpy as np
import pickle
import sys
import os.path
import tensorflow as tf
import copy
from vocab import *


class Datatype:
	
	def __init__(self, name, codes, lang1, lang2, title, num_samples, max_lang1, max_lang2, ls):

		self.name = name
		self.codes = codes
		self.lang1 = lang1
		self.lang2 = lang2
		self.title = title
		self.num_samples = num_samples
		self.max_lang1 = max_lang1
		self.max_lang2 = max_lang2
		self.global_count_train = 0
		self.global_count_test = 0
		self.ls = ls


class PadDataset:
	
	def pad_data(self, data, max_length):

		padded_data = []

		for lines in data:
			if (len(lines) < max_length):
				temp = np.lib.pad(lines, (0,max_length - len(lines)),
					'constant', constant_values=0)
			else:
				temp = lines[:max_length]
			padded_data.append(temp)

		return padded_data







	def make_batch(self, data, ls, batch_size, count, max_length):

		batch = []
		batch = [data[code] for code in ls[count:count+batch_size]]
		count += batch_size

		while (len(batch)<batch_size):
			batch.append(np.zeros(max_length, dtype = int))
			count = 0

		batch = self.pad_data(batch, max_length)
		batch = np.transpose(batch)
		return batch, count	 


	def next_batch(self, dt, batch_size, c=True):

		if c is True:
			count = dt.global_count_train
		else:
			count = dt.global_count_test

		max_lang1 = max(val.max_lang1 for i, val in self.datasets.iteritems())
		max_lang2 = max(val.max_lang2 for i, val in self.datasets.iteritems())
		print max_lang1

		lang1_data, count1 = self.make_batch(dt.lang1, dt.ls, batch_size, count, max_lang1)
		lang2_data, _ = self.make_batch(dt.lang2, dt.ls, batch_size, count, max_lang2)
		title_data, _ = self.make_batch(dt.title, dt.ls, batch_size, count, max_lang2)

		if (c == True): 
			dt.global_count_train = count1 % dt.num_samples
		else:
			dt.global_count_test = count1 % dt.num_samples

		weights = copy.deepcopy(title_data)    

		for i in range(len(title_data)):
			for j in range(len(title_data[0])):
				if (weights[i][j] > 0):
						weights[i][j] = 1
				else:
						weights[i][j] = 0


		return lang1_data, lang2_data, title_data, weights, max_lang1, max_lang2    





	def load_data_file(self, name, lang1_file, lang2_file, ls):
		
				
		f = open(lang1_file, 'r').read()
		f = f.split('\n')
		g = open(lang2_file, 'r').read()
		g = g.split('\n')
		
		d1 = {}
		d2 = {}
		title = {}
		max_lang1 = 0
		for line in f:
			try:
				code = line.split(':')[0]
				if code in ls:
					x = line.split(':')[1]
					lword  = x.split('?')[1][:-1]
					temp = [self.vocab.encode_word(i) for i in unicode(lword, 'utf-8')]
					if (len(temp)> max_lang1):
						max_lang1 = len(temp)
					d1[code] = temp
			except IndexError:
				print 'IndexError'				
		max_lang2 = 0				
		for line in g:
			try:
				code = line.split(':')[0]
				if code in ls:
					x = line.split(':')[1]
					lword  = x.split('?')[1][:-1]
					#print lword
					temp = [self.vocab.encode_word(i) for i in unicode(lword, 'utf-8')]
					if (len(temp)> max_lang2):
						max_lang2 = len(temp)
					temp.insert(0, self.vocab.encode_word('<s>'))	
					temp.append(self.vocab.encode_word('<eos>'))	
					title[code] = temp[:-1]
					d2[code] = temp[1:]
			except IndexError:
				print 'IndexError'		


		return Datatype(name, ls, d1, d2, title, len(ls), max_lang1, max_lang2, ls)

	def load_data(self, lang1_file, lang2_file):

		self.datasets = {}
		def intersect(a, b):
			return list(set(a) & set(b))
		
		f = open(lang1_file, 'r').read()
		f = f.split('\n')
		ls1 = []
		for line in f:
			ls1.append(line.split(':')[0])


		g = open(lang2_file, 'r').read()
		g = g.split('\n')
		ls2 = []
		for line in g:
			ls2.append(line.split(':')[0])

		ls = intersect(ls1, ls2)
		try:
			ls.remove('')
		except:
			pass	
		l = len(ls)
		tr_l = int(l*0.8)
		te_l = int(l*0.1)
		self.datasets[0] = self.load_data_file('train', lang1_file, lang2_file, ls[:tr_l])
		self.datasets[1] = self.load_data_file('test', lang1_file, lang2_file, ls[tr_l:tr_l+te_l])
		self.datasets[2] = self.load_data_file('train', lang1_file, lang2_file, ls[tr_l+te_l:])


	def __init__(self, lang1_file, lang2_file, embedding_size = 100, global_count = 0):

		filenames = [lang1_file, lang2_file]

		self.global_count = 0
		self.vocab = Vocab()
		self.vocab.construct_vocab(filenames, embedding_size)
		self.load_data(lang1_file, lang2_file)

	def length_vocab(self):
		
		return self.vocab.len_vocab

	def decode_to_sentence(self, decoder_states):

		s =''
		for state in decoder_states:
			word = ''
			if state not in self.vocab.index_to_word:
				word = '<unk>'
			else:
				word = self.vocab.index_to_word[state]
			s = s+word
			
		return s.encode('UTF-8')		

def main():
	paddata = PadDataset('new/bg_data.txt', 'new/ko_data.txt', embedding_size = 20)

	print paddata.datasets[2].lang2






if __name__ == '__main__':
	main()			 			

		
					

								
					

				 