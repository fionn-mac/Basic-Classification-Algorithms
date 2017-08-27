#!/usr/bin/env python

import sys
import os
import numpy as np
from operator import itemgetter

"""Feel free to add any extra classes/functions etc as and when needed.
This code is provided purely as a starting point to give you a fair idea
of how to go about implementing machine learning algorithms in general as
a part of the first assignment. Understand the code well"""


class FeatureVector(object):
	def __init__(self, vocabsize, numdata):
		self.vocabsize = vocabsize
		self.X =  []
		self.Y =  []

	def make_featurevector(self, input, classid):
		"""
		Takes input the documents and outputs the feature vectors as X and classids as Y.
		"""
		self.X.append({})
		self.Y.append(classid)
		i = len(self.Y) - 1
		
		# filter(lambda a: a != "<s>", input)
		# filter(lambda a: a != "<\s>", input)
		
		count = len(input)
		
		for word in input:
			if word not in self.X[i]:
				self.X[i][word] = float(0)
			self.X[i][word] += float(1)/float(count)

class KNN(object):
	def __init__(self, trainVec, testVec):
		self.X_train = trainVec.X
		self.Y_train = trainVec.Y
		self.X_test = testVec.X
		self.Y_test = testVec.Y
		self.metric = Metrics('accuracy')

	def classify(self, nn):
		"""
		Takes input X_train, Y_train, X_test and Y_test and displays the accuracies.
		"""
		
		correct = 0
		for j, fileStat in enumerate(self.X_test):
			valueMap = {}
			for i, trainStat in enumerate(self.X_train):
				valueMap[i] = [0, self.Y_train[i]]
				for word in trainStat:
					if word in fileStat:
						valueMap[i][0] += abs(trainStat[word] - fileStat[word])
					else:
						valueMap[i][0] += trainStat[word]

				for word in fileStat:
					if word not in trainStat:
						valueMap[i][0] += fileStat[word]

			maxCount = 0
			predictClass = 0
			count = {}
			temp = sorted(valueMap.items(), key=itemgetter(1))
			
			for key in temp[:nn]:
				if key[1][1] not in count:
					count[key[1][1]] = 0
				count[key[1][1]] += 1
				
				if count[key[1][1]] > maxCount:
					maxCount = count[key[1][1]]
					predictClass = key[1][1]
			
			# print "Predicted Class: ", predictClass, "Actual Class: ", self.Y_test[j]
			if predictClass == self.Y_test[j]:
				correct += 1

		print "Accuracy :", float(correct)*100/float(j+1), "%  with K: ", nn

class Metrics(object):
	def __init__(self,metric):
		self.metric = metric

	def score(self):
		if self.metric == 'accuracy':
			return self.accuracy()
		elif self.metric == 'f1':
			return self.f1_score()

	def get_confmatrix(self,y_pred,y_test):
		"""
		Implements a confusion matrix
		"""

	def accuracy(self):
		"""
		Implements the accuracy function
		"""

	def f1_score(self):
		"""
		Implements the f1-score function
		"""

if __name__ == '__main__':
	datadir = './datasets/q4_1/'
	classes = ['galsworthy/', 'galsworthy_2/', 'mill/', 'shelley/', 'thackerey/', 'thackerey_2/', 'wordsmith_prose/', 'cia/', 'johnfranklinjameson/', 'diplomaticcorr/']
	inputdir = ['train/', 'test/']

	vocab = 1000
	trainsz = 1000
	testsz = 1000

	print('Making the feature vectors.')
	trainVec = FeatureVector(vocab, trainsz)
	testVec = FeatureVector(vocab, testsz)

	for idir in inputdir:
		classid = 1
		for c in classes:
			listing = os.listdir(datadir+idir+c)
			for filename in listing:
				f = open(datadir+idir+c+filename,'r')
				
				inputs = []
				for line in f:
					inputs.extend(line.split())

				if idir == 'train/':
					trainVec.make_featurevector(inputs, classid)
				else:
					testVec.make_featurevector(inputs, classid)
			classid += 1

	print('Finished making features.')
	print('Statistics ->')
	# print(trainVec.X.shape, trainVec.Y.shape, testVec.X.shape, testVec.Y.shape)

	knn = KNN(trainVec, testVec)
	
	for i in xrange(10):
		knn.classify(i+1)
