"""
File: grader.py
-----------------------------
Milestone 4 answer:
以下為各 epoch 最後的 error ->

epoch   train   validation
 40     0.036     0.267
 70     0.014     0.261
 80     0.012     0.263
 90     0.009     0.265

=> 發現epoch越大，train error越小但validation error並沒有一起變小 ==> 發生overfitting！！！
反而在70的 validation error 最小

"""


# !/usr/bin/python3

import graderUtil
import util
import time
from util import *

grader = graderUtil.Grader()
submission = grader.load('submission')
util = grader.load('util')


############################################################
# Milestone 3: Sparse Vector
############################################################

### 3a

# Basic sanity check for feature extraction
def test3a0():
    ans = {"a":2, "b":1}
    grader.require_is_equal(ans, submission.extractWordFeatures("a b a"))
grader.add_basic_part('3a-0', test3a0, max_points=10, max_seconds=1, description="basic test")

### 3b

def test3b0():
    ans = {'a': 1.1, 'b': 1, 'c': 1.1, 'd':0.1}
    d1 = {'a':1, 'b':1, 'c':1}
    d2 = {'a':1, 'c':1, 'd':1}
    util.increment(d1, 0.1, d2)
    grader.require_is_equal(ans, d1)
grader.add_basic_part('3b-0', test3b0, max_points=10, max_seconds=1, description="basic sanity check for adding scale * d2 on d1")

### 3c

def test3c0():
    ans = 0.05
    grader.require_is_equal(ans, util.dotProduct({'Movie': 0.1, 'is': 0.2, 'good': 0.25},  {'Movie': 0.1, 'is': 0.2, 'very': -0.25, 'bad': -0.25}))
grader.add_basic_part('3c-0', test3c0, max_points=10, max_seconds=1, description="basic sanity check for the answer of dot product of 2 sparse vectors d1 and d2")


############################################################
# Milestone 4: Sentiment Classification
############################################################

def test4a0():
    trainExamples = (("hello world", 1), ("goodnight moon", -1))
    testExamples = (("hello", 1), ("moon", -1))
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(trainExamples, testExamples, featureExtractor, numEpochs=40, alpha=0.01)
    grader.require_is_greater_than(0, weights["hello"])
    grader.require_is_less_than(0, weights["moon"])
grader.add_basic_part('4a-0', test4a0, max_points=5, max_seconds=1, description="basic sanity check for learning correct weights on two training and testing examples each")

def test4a1():
    trainExamples = (("hi bye", 1), ("hi hi", -1))
    testExamples = (("hi", -1), ("bye", 1))
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(trainExamples, testExamples, featureExtractor, numEpochs=40, alpha=0.01)
    grader.require_is_less_than(0, weights["hi"])
    grader.require_is_greater_than(0, weights["bye"])
grader.add_basic_part('4a-1', test4a1, max_points=5, max_seconds=2, description="test correct overriding of positive weight due to one negative instance with repeated words")

def test4a2():
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = submission.extractWordFeatures
    # featureExtractor = submission.extractCharacterFeatures(4)
    weights = submission.learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=40, alpha=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) > 0 else -1))
    validationError = evaluatePredictor(validationExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) > 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))
    grader.require_is_less_than(0.04, trainError)
    grader.require_is_less_than(0.30, validationError)
grader.add_basic_part('4a-2', test4a2, max_points=30, max_seconds=45, description="test classifier on real polarity dev dataset")


############################################################
# Milestone 5: Finishing up
# TODO: Is there any difference between using extractCharacterFeatures and extractWordFeatures? Why?
# TODO: Your answer here
# TODO: 在一樣是numEpoch＝40的狀況下，首先感受到的不同是速度比較慢，再來是最後的training error & validation error的差異也是因為 overfitting
#                   training error    validation error
#       extractC      0.002            0.311 (>0.300 FAIL)
#       extractW      0.036            0.267
#
############################################################

### 5a

def test5a0():
    weights = {"hello": 1, "world": 1}
    data = submission.generateDataset(5, weights)
    for datapt in data:
        grader.require_is_equal((util.dotProduct(datapt[0], weights) >= 0), (datapt[1] == 1))
grader.add_basic_part('5a-0', test5a0, max_points=5, max_seconds=2, description="test correct generation of random dataset labels")


def test5a1():
    weights = {}
    for i in range(100):
        weights[str(i + 0.1)] = 1
    data = submission.generateDataset(100, weights)
    for datapt in data:
        grader.require_is_equal(False, dotProduct(datapt[0], weights) == 0)
grader.add_basic_part('5a-1', test5a1, max_points=5, max_seconds=2, description="test that the randomly generated example actually coincides with the given weights")

### 5b

def test5b0():
    fe = submission.extractCharacterFeatures(3)
    sentence = "hello world hel"
    ans = {"hel":2, "ell":1, "llo":1, "low":1, "owo":1, "wor":1, "orl":1, "rld":1, "ldh":1, "dhe":1}
    grader.require_is_equal(ans, fe(sentence))
grader.add_basic_part('5b-0', test5b0, max_points=20, max_seconds=1, description="test basic character n-gram features")


grader.grade()
