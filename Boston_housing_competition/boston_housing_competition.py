"""
File: boston_housing_competition.py
Name: Linda
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientist!
"""

import pandas as pd
from sklearn import metrics, linear_model, preprocessing, decomposition

TRAIN_DATA = 'boston_housing/train.csv'
TEST_DATA = 'boston_housing/test.csv'


def main():
	# Read file
	data = pd.read_csv(TRAIN_DATA)

	# Check null
	# print(data.isnull().sum())

	# Extract true labels
	labels = data.medv

	# Extract features
	training_data = data.drop(columns=['ID', 'zn', 'indus', 'medv'])

	# Standardization
	standardizer = preprocessing.StandardScaler()
	training_data = standardizer.fit_transform(training_data)

	# Training
	h = linear_model.LinearRegression()
	classifier = h.fit(training_data, labels)
	acc = classifier.score(training_data, labels)
	print('Accuracy:', acc)

	# Test
	test_data = pd.read_csv(TEST_DATA)

	# Check null
	# print(test_data.isnull().sum())

	# Extract features
	id_lst = list(test_data.ID)
	testing_data = test_data.drop(columns=['ID', 'zn', 'indus'])


	# Standardization
	testing_data = standardizer.transform(testing_data)

	# Predict
	predictions = classifier.predict(testing_data)
	out_file(predictions, 'pandas_boston_ver1.csv', id_lst)


def out_file(predictions, filename, id_lst):
	print('\n===============================')
	print(f'Writing predictions to -> {filename}')
	with open(filename, 'w') as f_out:
		f_out.write('ID,medv\n')
		i = 0
		for ans in predictions:
			f_out.write(str(id_lst[i])+','+str(ans)+'\n')
			i += 1
	print('===============================')



















if __name__ == '__main__':
	main()
