#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
import copy
import math
from utils import *

class LanguageModel(object):
    def __init__(self, corpus, ngram, min_freq, uniform=False):
        """
        Initialize language model
        :param corpus: input text corpus to build LM on
        :param ngram: number of n-gram, e.g. 1, 2, 3, ...
        :param min_freq: minimum frequency threshold to set a word to UNK placeholder
                         set to 1 to not use this threshold
        :param uniform: boolean flag, set to True to indicate this model is a simple uniform LM
                        otherwise will be an N-gram model
        """
        self.min_freq = min_freq
        self.corpus = corpus
        self.n = ngram
        self.unif = uniform
        self.lm = self.build()
                
    def build(self):
        """
        Build LM from text corpus
        :return: dictionary LM
        """
        corpus = copy.deepcopy(self.corpus)
        if self.min_freq > 1:
            tokens = {}
            for line in corpus:
                for word in line:
                    if word in tokens.keys(): tokens[word] += 1
                    else: tokens[word] = 1
            for key in tokens.keys():
                if tokens[key] < self.min_freq: 
                    for i in range(len(corpus)):
                        for j in range(len(corpus[i])):
                            if key == corpus[i][j]: corpus[i][j] = 'UNK'
        tokens = {}
        if self.unif:
            for line in corpus:
                for word in line:
                    if not (word in tokens.keys()): tokens[word] = 1
            return tokens
        lm = {}
        for line in corpus:
            for i in range(len(line) - self.n + 1):
                if self.n == 1:
                    key = line[i]
                    if key in lm.keys(): lm[key] += 1
                    else: lm[key] = 1
                else:
                    key2 = line[i + self.n - 1]
                    key1 = tuple(line[i:i + self.n - 1])
                    if key1 in lm.keys(): 
                        if key2 in lm[key1].keys():
                            lm[key1][key2] += 1
                        else: lm[key1][key2] = 1
                    else:
                        lm[key1] = {}
                        lm[key1][key2] = 1
        return lm

    def most_common_words(self, k):
        """
        This function will only be called after the language model has been built
        Your return should be sorted in descending order of frequency
        Sort according to ascending alphabet order when multiple words have same frequency
        :return: list[tuple(token, freq)] of top k most common tokens
        """
        if self.unif or self.n == 1:
            dlist = self.lm.items()
            dlist = sorted(dlist, key = lambda tup: (-tup[1], tup[0].lower()))
            return dlist[:k]
        else:
            dlist = []
            for key1 in self.lm.keys():
                for key2 in self.lm[key1].keys():
                    string = ''
                    for s in key1: string += s + ' '
                    string += key2
                    dlist.append((string, self.lm[key1][key2]))
            dlist = sorted(dlist, key = lambda tup: (-tup[1], tup[0].lower()))
            return dlist[:k]

def calculate_perplexity(models, coefs, data):
    """
    Calculate perplexity with given model
    :param models: language models
    :param coefs: coefficients
    :param data: test data
    :return: perplexity
    """
    newdata = copy.deepcopy(data)
    min_freq = max(models[0].min_freq, models[1].min_freq, models[2].min_freq, models[3].min_freq)
    if min_freq > 1:
        words = {}
        for line in newdata:
            for token in line:
                words[token] = words[token] + 1 if token in words else 1
        for i in range(len(newdata)):
            for j in range(len(newdata[i])):
                if words[newdata[i][j]] < min_freq: newdata[i][j] = 'UNK'
                if not (newdata[i][j] in models[0].lm.keys()): newdata[i][j] = 'UNK'
    unif = models[0].lm
    v = len(unif.keys())
    uni = models[1].lm
    n = sum(list(uni.values()))
    bi = models[2].lm
    bidict = {}
    for key in bi.keys():
        bidict[key] = sum(list(bi[key].values()))
    tri = models[3].lm
    tridict = {}
    for key in tri.keys():
        tridict[key] = sum(list(tri[key].values()))
    total = 0
    numwords = 0
    for line in newdata:
        numwords += len(line)
        for i in range(len(line)):
            p_unif = coefs[0] / float(v)
            if line[i] in uni: p_uni = coefs[1] * (uni[line[i]]+1) / float(n+v)
            else: p_uni = p_unif
            if (line[i-1],) in bi and line[i] in bi[(line[i-1],)] and i > 0:
                p_bi = coefs[2] * bi[(line[i-1],)][line[i]] / float(bidict[(line[i-1],)])
            elif (line[i-1],) in bi and i > 0: p_bi = coefs[2] / float(bidict[(line[i-1],)])
            else: p_bi = coefs[2] * min(1/ float(v), p_unif, p_uni)
            if (line[i-2],line[i-1]) in tri and line[i] in tri[(line[i-2],line[i-1])] and i > 1:
                p_tri = coefs[3] * tri[(line[i-2],line[i-1])][line[i]] / float(tridict[(line[i-2],line[i-1])])
            elif (line[i-2],line[i-1]) in tri and i > 1:
                p_tri = coefs[3] / float(tridict[(line[i-2],line[i-1])])
            else: p_tri = coefs[3] * min(1 / v, p_unif, p_uni, p_bi)
            if not (p_unif == 0 and p_uni == 0 and p_bi == 0 and p_tri == 0):
                total += math.log2(p_unif + p_uni + p_bi + p_tri)
    return math.pow(2, total / float(-numwords))

# Do not modify this function!
def parse_args():
    """
    Parse input positional arguments from command line
    :return: args - parsed arguments
    """
    parser = argparse.ArgumentParser('N-gram Language Model')
    parser.add_argument('coef_unif', help='coefficient for the uniform model.', type=float)
    parser.add_argument('coef_uni', help='coefficient for the unigram model.', type=float)
    parser.add_argument('coef_bi', help='coefficient for the bigram model.', type=float)
    parser.add_argument('coef_tri', help='coefficient for the trigram model.', type=float)
    parser.add_argument('min_freq', type=int,
                        help='minimum frequency threshold for substitute '
                             'with UNK token, set to 1 for not use this threshold')
    parser.add_argument('testfile', help='test text file.')
    parser.add_argument('trainfile', help='training text file.', nargs='+')
    args = parser.parse_args()
    return args

# Main executable script provided for your convenience
# Not executed on autograder, so do what you want
if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # load and preprocess train and test data
    train = preprocess(load_dataset(args.trainfile))
    test = preprocess(read_file(args.testfile))

    # build language models
    uniform = LanguageModel(train, ngram=1, min_freq=args.min_freq, uniform=True)
    unigram = LanguageModel(train, ngram=1, min_freq=args.min_freq)
    bigram = LanguageModel(train, ngram=2, min_freq=args.min_freq)
    trigram = LanguageModel(train, ngram=3, min_freq=args.min_freq)

    # calculate perplexity on test file

    ppl = calculate_perplexity(
        models=[uniform, unigram, bigram, trigram],
        coefs=[args.coef_unif, args.coef_uni, args.coef_bi, args.coef_tri],
        data=test)

    print("Perplexity: {}".format(ppl))