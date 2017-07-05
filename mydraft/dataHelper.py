#-*- coding:utf-8 -*-

import os
import sys
import numpy as np
from numpy import *
import gensim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import preprocessing

dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))

class LoadData(object):
    """docstring for LoadData"""
    def __init__(self,filename,count=1000):
        super(LoadData, self).__init__()
        self.model = gensim.models.Word2Vec.load(dirname+"/word2vec/wiki.ch.text.model")
        self.fname=filename
        if filename == 'ch_waimai2_corpus.txt':
            count=1000
        self.pos_count=count
        self.neg_count=count
        # self.enc = preprocessing.OneHotEncoder()

    def getWordVector(self,word,size=400):
        vec = np.zeros(size).reshape(1,size)
        print word
        vec = self.model[word]
    def buildWordVector(self,words,size=400):
        vec = np.zeros(size).reshape((1,size))
        count = 0
        for word in words:
            try:
                vec += self.model[word].reshape((1,size))
                count += 1
            except KeyError:
                continue
        if count != 0:
            vec /= count
        return vec
    def train_data(self):
        path = os.path.join(dirname,"testData")
        #fname = "ch_waimai2_corpus.txt"
        # fname = "ch_hotel_corpus.txt"
        infile = open(path + "/"+self.fname)
        posD=[]
        negD=[]
        rdata1=[]
        rdata2=[]
        for line in infile.readlines():
            if len(line)<4:
                pass
            if line[0:3] == "neg":
                negD.append(line[4:].strip().strip("\n"))
            else:
                posD.append(line[4:].strip().strip("\n"))

        print "pos counts:",len(posD)
        print "neg counts:",len(negD)
        if len(posD)<self.pos_count or len(negD)<self.neg_count:
        	self.pos_count=len(posD)
        	self.neg_count=len(negD)


        if self.pos_count > 0 :
            shuffleArray = range(len(posD))
            np.random.shuffle(shuffleArray)
            for ii in xrange(self.pos_count):
                rdata1.append(posD[shuffleArray[ii]])
        else:
            rdata1 = posD

        if self.neg_count > 0:
            shuffleArray = range(len(negD))
            np.random.shuffle(shuffleArray)
            for ii in xrange(self.neg_count):
                rdata2.append(negD[shuffleArray[ii]])
        else:
            rdata2 = negD

        y = np.concatenate((np.ones(len(rdata1)), np.zeros(len(rdata2))))
        X = np.concatenate((rdata1, rdata2))

        X = np.concatenate([self.buildWordVector(z) for z in X])
        X = scale(X)
        X_vec = []
        for item in X:
            X_vec.append(tuple(item.tolist()))

        self.pos = X_vec[0:len(rdata1)]
        self.neg = X_vec[0:len(rdata2)]
        print "#################",len(rdata2)
        X_train,X_test,y_train,y_test = train_test_split(X_vec,y,test_size=0.2)


        X_train=np.array(X_train)

        # y_train=np.array(y_train)
        # y_train=y_train.reshape(len(y_train),1)

        y_train=self.trans_ont_hot(y_train)

        X_test=np.array(X_test)

        # y_test=np.array(y_test)
        # y_test=y_test.reshape(len(y_test),1)
        y_test=self.trans_ont_hot(y_test)

        self.X_test=X_test
        self.y_test=y_test


        return  (X_train,y_train)
    def trans_ont_hot(self,ls,c=2):
        result=np.zeros((len(ls),c))
        for index in range(len(ls)):
            result[index][ls[index]]=1
        return result
    def test_data(self):
        return (self.X_test,self.y_test)
    def next_batch(self,count=50):
    	print "#####################"
    	print self.pos_count,self.neg_count
    	print len(self.pos),len(self.neg)
        shuffleArray = range(self.pos_count)
        np.random.shuffle(shuffleArray)

        pos=[]
        neg=[]
        for ii in xrange(count/2):
            pos.append(self.pos[shuffleArray[ii]])
        shuffleArray = range(self.neg_count)
        np.random.shuffle(shuffleArray)
        for ii in xrange(count/2):
            neg.append(self.neg[shuffleArray[ii]])

        y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
        X = np.concatenate((pos, neg))
        X=np.array(X)

        # y=np.array(y)
        # y=y.reshape(len(y),1)
        y=self.trans_ont_hot(y)

        return  (X,y)


