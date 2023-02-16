#!/usr/bin/python

import math
import random
from collections import defaultdict
from util import *
from typing import Any, Dict, Tuple, List, Callable

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


############################################################
# Milestone 3a: feature extraction

def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    d = defaultdict(int)  # default value 0
    for ch in x.split():
        d[ch] += 1
    return d
    # END_YOUR_CODE


############################################################
# Milestone 4: Sentiment Classification

def learnPredictor(trainExamples: List[Tuple[Any, int]], validationExamples: List[Tuple[Any, int]],
                   featureExtractor: Callable[[str], FeatureVector], numEpochs: int, alpha: float) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement gradient descent.
    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. Note also that the 
    identity function may be used as the featureExtractor function during testing.
    """
    weights = {}  # feature => weight
    training_data = []

    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)

    def predictor(str_x):
        predict_k = dotProduct(weights, featureExtractor(str_x))
        return 1 if predict_k >= 0 else -1

    for epoch in range(numEpochs):
        cost = 0
        for x, y in trainExamples:
            y = 0 if y == -1 else 1
            x_dict = featureExtractor(x)
            k = dotProduct(weights, x_dict)
            h = 1/(1+math.exp(-k))   # sigmoid(k)
            loss = -(y*math.log(h)+(1-y)*math.log(1-h))
            cost += loss
            # SGD for each weight
            increment(weights, -alpha*(h-y), featureExtractor(x))

        print(f'Training Error: ({epoch} epoch):{evaluatePredictor(trainExamples, predictor)}')
        print(f'Validation Error: ({epoch} epoch):{evaluatePredictor(validationExamples, predictor)}')

    # for tup in trainExamples:  # 把 trainExamples 每個tuple拿出來
    #     mutable_tup = list(tup)  # 把 tuple變成 list
    #     mutable_tup[1] = 0 if mutable_tup[1] == -1 else 1  # -1 變 0
    #
    #     y, comment = mutable_tup[1], mutable_tup[0]
    #     feature_vector = featureExtractor(comment)  # 把文字轉成dict(紀錄出現了什麼字，然後次數是多少）
    #     training_data.append((feature_vector, y))
    #
    # for epoch in range(numEpochs):
    #     cost = 0
    #     for x, y in training_data:
    #         k = dotProduct(weights, x)
    #         h = 1/(1+math.exp(-k))   # sigmoid(k)
    #         loss = -(y*math.log(h)+(1-y)*math.log(1-h))
    #         cost += loss
    #         # SGD for each weight
    #         increment(weights, -alpha*(h-y), x)
    #
    #     print(f'Training Error: ({epoch} epoch):{evaluatePredictor(training_data, predictor)}')
    #     print(f'Validation Error: ({epoch} epoch):{evaluatePredictor(validationExamples, predictor)}')

    # END_YOUR_CODE
    return weights


############################################################
# Milestone 5a: generate test case

def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    def generateExample() -> Tuple[Dict[str, int], int]:
        """
        Return a single example (phi(x), y).
        phi(x) should be a dict whose keys are a subset of the keys in weights
        and values are their word occurrance.
        y should be 1 or -1 as classified by the weight vector.
        Note that the weight vector can be arbitrary during testing.
        """
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        len_phi = random.randint(1, len(weights))  # 隨機feature vector的長度 = 該評論的字數
        phi = defaultdict(int)
        weights_words = []

        for key in weights:
            weights_words.append(key)

        for i in range(len_phi):
            word_id = random.randint(0, len(weights_words)-1)  # 隨機選一個要選的字的號碼
            token = weights_words[word_id]  # 取出那個字
            phi[token] += random.randint(1, len_phi)   # 放入phi這個dictionary並給他一個隨機的 occurrance
            # weights_words.remove(token)  # 怕重複所以要把用過的字刪掉

        ans = dotProduct(weights, phi)  # 計算設計出來隨機的phi(feature vector)屬於好影評還是壞影評

        y = 1 if ans >= 0 else -1  # 紀錄真實值

        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Milestone 5b: character features

def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        d_char = {}
        new_x = x.replace(' ', '')
        for start in range(len(new_x)):
            if start <= (len(new_x) - n):
                char_key = new_x[start:start + 3]
                if char_key not in d_char:
                    d_char[char_key] = 1
                else:
                    d_char[char_key] += 1
        return d_char
        # END_YOUR_CODE
    return extract


############################################################
# Problem 3f: 
def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    """
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=20, alpha=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples,
                                   lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(validationExamples,
                                        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))

