"""
File: titanic_level1.py
Name: 
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python codes. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle. This model is the most flexible one among all
levels. You should do hyperparameter tuning and find the best model.
"""

import math
import util
from util import *

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: an empty Python dictionary
	:param mode: str, indicating the mode we are using
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""
	if mode == 'Train':
		is_head = True
		with open(filename, 'r') as f:
			for line in f:
				if is_head:
					key = line.strip()
					key_list = key.split(',')
					for k in range(len(key_list)):
						if k == 1:
							data[key_list[k]] = list()
						if k == 2:
							data[key_list[k]] = list()
						if k == 4:
							data[key_list[k]] = list()
						if k == 5:
							data[key_list[k]] = list()
						if k == 6:
							data[key_list[k]] = list()
						if k == 7:
							data[key_list[k]] = list()
						if k == 9:
							data[key_list[k]] = list()
						if k == 11:
							data[key_list[k]] = list()
					is_head = False
				else:
					line = line.strip()
					value_list = line.split(',')
					for i in range(len(value_list)):
						if not value_list[6]:
							break
						elif not value_list[12]:
							break
						if i == 1:  # Survived
							data['Survived'].append(int(value_list[i]))
						if i == 2:  # Pclass
							data['Pclass'].append(int(value_list[i]))
						if i == 5:  # Sex
							if value_list[i] == 'male':
								sex_value = 1
							else:
								sex_value = 0
							data['Sex'].append(sex_value)
						if i == 6:  # Age
							data['Age'].append(float(value_list[i]))
						if i == 7:  # SibSp
							data['SibSp'].append(int(value_list[i]))
						if i == 8:  # Parch
							data['Parch'].append(int(value_list[i]))
						if i == 10:  # Fare
							data['Fare'].append(float(value_list[i]))
						if i == 12:  # Embarked
							if value_list[i] == 'S':
								embark_value = 0
							elif value_list[i] == 'C':
								embark_value = 1
							else:
								embark_value = 2
							data['Embarked'].append(embark_value)
	else:
		is_head = True
		with open(filename, 'r') as f:
			for line in f:
				if is_head:
					key = line.strip()
					key_list = key.split(',')
					for k in range(len(key_list)):
						if k == 1:
							data[key_list[k]] = list()
						if k == 3:
							data[key_list[k]] = list()
						if k == 4:
							data[key_list[k]] = list()
						if k == 5:
							data[key_list[k]] = list()
						if k == 6:
							data[key_list[k]] = list()
						if k == 8:
							data[key_list[k]] = list()
						if k == 10:
							data[key_list[k]] = list()
					is_head = False
				else:
					line = line.strip()
					value_list = line.split(',')
					for i in range(len(value_list)):
						if i == 1:  # Pclass
							data['Pclass'].append(int(value_list[i]))
						if i == 4:  # Sex
							if value_list[i] == 'male':
								sex_value = 1
							else:
								sex_value = 0
							data['Sex'].append(sex_value)
						if i == 5:  # Age
							if not value_list[i]:
								fill_age_na = (sum(training_data['Age']) / len(training_data['Age']))
								fill_age_na = round(fill_age_na, 3)
								value_list[i] = fill_age_na
							data['Age'].append(float(value_list[i]))
						if i == 6:  # SibSp
							data['SibSp'].append(int(value_list[i]))
						if i == 7:  # Parch
							data['Parch'].append(int(value_list[i]))
						if i == 9:  # Fare
							if not value_list[i]:
								fill_fare_na = (sum(training_data['Fare']) / len(training_data['Fare']))
								fill_fare_na = round(fill_fare_na, 3)
								value_list[i] = fill_fare_na
							data['Fare'].append(float(value_list[i]))
						if i == 11:  # Embarked
							if value_list[i] == 'S':
								embark_value = 0
							elif value_list[i] == 'C':
								embark_value = 1
							else:
								embark_value = 2
							data['Embarked'].append(embark_value)
	return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""
	if feature == 'Sex':
		data['Sex_0'] = list()
		data['Sex_1'] = list()
		for value in data[feature]:
			if value == 1:
				data['Sex_0'].append(0)
				data['Sex_1'].append(1)
			else:
				data['Sex_0'].append(1)
				data['Sex_1'].append(0)
		del data[feature]
	if feature == 'Pclass':
		data['Pclass_0'] = list()
		data['Pclass_1'] = list()
		data['Pclass_2'] = list()
		for value in data[feature]:
			if value == 1:
				data['Pclass_0'].append(1)
				data['Pclass_1'].append(0)
				data['Pclass_2'].append(0)
			elif value == 2:
				data['Pclass_0'].append(0)
				data['Pclass_1'].append(1)
				data['Pclass_2'].append(0)
			elif value == 3:
				data['Pclass_0'].append(0)
				data['Pclass_1'].append(0)
				data['Pclass_2'].append(1)
		del data[feature]
	if feature == 'Embarked':
		data['Embarked_0'] = list()
		data['Embarked_1'] = list()
		data['Embarked_2'] = list()
		for value in data[feature]:
			if value == 0:
				data['Embarked_0'].append(1)
				data['Embarked_1'].append(0)
				data['Embarked_2'].append(0)
			elif value == 1:
				data['Embarked_0'].append(0)
				data['Embarked_1'].append(1)
				data['Embarked_2'].append(0)
			elif value == 2:
				data['Embarked_0'].append(0)
				data['Embarked_1'].append(0)
				data['Embarked_2'].append(1)
		del data[feature]

	return data


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	for key, value in data.items():
		max_value = max(value)
		min_value = min(value)
		for i in range(len(value)):
			data[key][i] = (data[key][i] - min_value) / (max_value - min_value)
	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature => weight
	keys = list(inputs.keys())
	if degree == 1:
		for i in range(len(keys)):
			weights[keys[i]] = 0
	elif degree == 2:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0
	# Start training
	if degree == 1:
		feature_vectors = {}
		for epoch in range(num_epochs):
			for i in range(len(labels)):
				# Step 3 : Feature Extract
				for feature, passenger_list in inputs.items():
					feature_vectors[feature] = passenger_list[i]
				y = labels[i]
				k = util.dotProduct(weights, feature_vectors)
				h = 1/(1+math.exp(-k))
				# Step 4 : Update weights
				util.increment(weights, -alpha*(h-y), feature_vectors)
	elif degree == 2:
		feature_vectors = {}
		temp_feature_vectors = {}
		for epoch in range(num_epochs):
			for i in range(len(labels)):
				# Step 3 : Feature Extract
				# degree 1
				for feature, passenger_list in inputs.items():
					feature_vectors[feature] = passenger_list[i]
				# degree 2
				for t in range(len(keys)):
					for j in range(len(keys)):
						if j >= t:
							temp_feature_vectors[keys[t]+keys[j]] = feature_vectors[keys[t]]*feature_vectors[keys[j]]
				feature_vectors.update(temp_feature_vectors)
				y = labels[i]
				k = util.dotProduct(weights, feature_vectors)
				h = 1 / (1 + math.exp(-k))
				# Step 4 : Update weights
				util.increment(weights, -alpha * (h - y), feature_vectors)
	return weights
