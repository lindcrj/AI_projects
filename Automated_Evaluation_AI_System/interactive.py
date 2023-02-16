"""
File: interactive.py
Name: Linda
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""

from util import *
from submission import *
from collections import defaultdict


WEIGHT = 'weights'


def main():
	weights = defaultdict(int)
	with open(WEIGHT, 'r') as f:
		for line in f:
			x = line.split('\t')
			key = x[0]
			value = x[1].strip()
			weights[key] += float(value)

	interactivePrompt(extractWordFeatures, weights)


if __name__ == '__main__':
	main()



