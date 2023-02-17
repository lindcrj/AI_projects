"""
File: titanic_level2.py
Name: 
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle. Hyperparameters are hidden by the library!
This abstraction makes it easy to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'
			 data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)
	labels = None
	if mode == 'Train':
		data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
		data = data.dropna(axis=0, how='any')   # axis : 0為列,1為行
		labels = data.Survived
		data = data.drop(columns=['Survived'])
		# 轉成數字
		# Changing 'male' to 1, 'female' to 0
		data.loc[data.Sex == 'male', 'Sex'] = 1
		data.loc[data.Sex == 'female', 'Sex'] = 0

		# Changing 'S' to 0, 'C' to 1, 'Q' to 2
		data.loc[data.Embarked == 'S', 'Embarked'] = 0
		data.loc[data.Embarked == 'C', 'Embarked'] = 1
		data.loc[data.Embarked == 'Q', 'Embarked'] = 2

		return data, labels

	elif mode == 'Test':
		data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

		# 處理age空值
		age_mean = round(training_data.Age.mean(), 3)
		data['Age'].fillna(value=age_mean, inplace=True)


		# 處理Fare空值
		fare_mean = round(training_data.Fare.mean(), 3)
		data['Fare'].fillna(value=fare_mean, inplace=True)
		# fare_median = data['Fare'].dropna().median()

		# 轉成數字
		# Changing 'male' to 1, 'female' to 0
		data.loc[data.Sex == 'male', 'Sex'] = 1
		data.loc[data.Sex == 'female', 'Sex'] = 0

		# Changing 'S' to 0, 'C' to 1, 'Q' to 2
		data.loc[data.Embarked == 'S', 'Embarked'] = 0
		data.loc[data.Embarked == 'C', 'Embarked'] = 1
		data.loc[data.Embarked == 'Q', 'Embarked'] = 2

		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	if feature == 'Sex':
		data.loc[data.Sex == 1, 'Sex_0'] = int(0)
		data.loc[data.Sex == 1, 'Sex_1'] = int(1)
		data.loc[data.Sex == 0, 'Sex_0'] = int(1)
		data.loc[data.Sex == 0, 'Sex_1'] = int(0)
		data.pop('Sex')
	elif feature == 'Pclass':
		data.loc[data.Pclass == 1, 'Pclass_0'] = int(1)
		data.loc[data.Pclass == 1, 'Pclass_1'] = int(0)
		data.loc[data.Pclass == 1, 'Pclass_2'] = int(0)
		data.loc[data.Pclass == 2, 'Pclass_0'] = int(0)
		data.loc[data.Pclass == 2, 'Pclass_1'] = int(1)
		data.loc[data.Pclass == 2, 'Pclass_2'] = int(0)
		data.loc[data.Pclass == 3, 'Pclass_0'] = int(0)
		data.loc[data.Pclass == 3, 'Pclass_1'] = int(0)
		data.loc[data.Pclass == 3, 'Pclass_2'] = int(1)
		data.pop('Pclass')
	elif feature == 'Embarked':
		data.loc[data.Embarked == 0, 'Embarked_0'] = int(1)
		data.loc[data.Embarked == 0, 'Embarked_1'] = int(0)
		data.loc[data.Embarked == 0, 'Embarked_2'] = int(0)
		data.loc[data.Embarked == 1, 'Embarked_0'] = int(0)
		data.loc[data.Embarked == 1, 'Embarked_1'] = int(1)
		data.loc[data.Embarked == 1, 'Embarked_2'] = int(0)
		data.loc[data.Embarked == 2, 'Embarked_0'] = int(0)
		data.loc[data.Embarked == 2, 'Embarked_1'] = int(0)
		data.loc[data.Embarked == 2, 'Embarked_2'] = int(1)
		data.pop('Embarked')
	return data


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""
	standardizer = preprocessing.StandardScaler()
	data = standardizer.fit_transform(data)
	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy
	on degree1; ~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimals)
	TODO: real accuracy on degree1 -> ______0.80196629______
	TODO: real accuracy on degree2 -> ______0.83707865______
	TODO: real accuracy on degree3 -> ______0.87640449______
	"""
	# ׿叫 data_preprocess(TRAIN_FILE) 來處理 NaN
	data, labels = data_preprocess(TRAIN_FILE, mode='Train')
	y = labels
	# 對 ‘Sex’, ‘Pclass’, ‘Embarked’ 分別׿叫一次 one_hot_encoding( )
	data = one_hot_encoding(data, 'Sex')
	data = one_hot_encoding(data, 'Pclass')
	data = one_hot_encoding(data, 'Embarked')
	# 標準化
	standardizer = preprocessing.StandardScaler()
	data = standardizer.fit_transform(data)

	# Polynomial
	poly_fea_extractor_2 = preprocessing.PolynomialFeatures(degree=2)
	data_2 = poly_fea_extractor_2.fit_transform(data)

	poly_fea_extractor_3 = preprocessing.PolynomialFeatures(degree=3)
	data_3 = poly_fea_extractor_3.fit_transform(data)


	# Degree 1
	h = linear_model.LogisticRegression(max_iter=10000)
	classifier = h.fit(data, y)
	acc = round(classifier.score(data, y), 8)
	print('Degree 1 Training Acc:', acc)

	# Degree 2
	h = linear_model.LogisticRegression(max_iter=10000)
	classifier = h.fit(data_2, y)
	acc = round(classifier.score(data_2, y), 8)
	print('Degree 2 Training Acc:', acc)
	#
	# # Degree 3
	h = linear_model.LogisticRegression(max_iter=10000)
	classifier = h.fit(data_3, y)
	acc = round(classifier.score(data_3, y), 8)
	print('Degree 3 Training Acc:', acc)




if __name__ == '__main__':
	main()
