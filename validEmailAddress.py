"""
File: validEmailAddress.py
Name: Linda
----------------------------
This file shows what a feature vector is
and what a weight vector is for valid email 
address classifier. You will use a given 
weight vector to classify what is the percentage
of correct classification.

Accuracy of this model: 0.6153846153846154 TODO:
"""



WEIGHT = [                           # The weight vector selected by Jerry
	[0.4],                           # (see assignment handout for more details)
	[0.4],
	[0.2],
	[0.2],
	[0.9],
	[-0.65],
	[0.1],
	[0.1],
	[0.1],
	[-0.1]
]

DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed


def main():
	scores = []
	valid = 0
	ind = -1

	maybe_email_list = read_in_data(DATA_FILE)

	# print(maybe_email_list)
	for maybe_email in maybe_email_list:
		ind += 1
		feature_vector = feature_extractor(maybe_email)
		# print(feature_vector)
		sum = 0
		for i in range(len(WEIGHT)):
			sum += feature_vector[i] * WEIGHT[i][0]
		scores.append(sum)

		if maybe_email_list.index(maybe_email) <= 12:
			if sum <= 0:
				valid += 1
		else:
			if sum > 0:
				valid += 1
	accu = valid / 26
	print('Accuracy: ', accu)
	# TODO:


def feature_extractor(maybe_email):
	"""
	:param maybe_email: str, the string to be processed
	:return: list, feature vector with 10 values of 0's or 1's
	"""
	feature_vector = [0] * len(WEIGHT)
	for i in range(len(feature_vector)):
		if i == 0:
			feature_vector[i] = 1 if '@' in maybe_email else 0
		elif i == 1:
			if feature_vector[0]:  # if feature_vector[0] = 1 確認email中包含＠
				feature_vector[i] = 1 if '.' not in maybe_email.split('@')[0] else 0
				# maybe_email.split('@')[0] => 因為spilt會把＠前後分成list的兩個element, 所以只看前面就是取[0]
		elif i == 2:
			if feature_vector[0]:
				feature_vector[i] = 1 if maybe_email.split('@')[0] is not '' else 0
		elif i == 3:
			if feature_vector[0]:
				feature_vector[i] = 1 if maybe_email.split('@')[1] is not '' else 0
		elif i == 4:   # 找最後一個
			if feature_vector[0]:
				target = maybe_email.rfind('@')
				tar_str = maybe_email[target:]
				feature_vector[i] = 1 if '.' in tar_str else 0
		elif i == 5:
			feature_vector[i] = 1 if ' ' not in maybe_email else 0
		elif i == 6:
			feature_vector[i] = 1 if maybe_email.endswith('.com') else 0
		elif i == 7:
			feature_vector[i] = 1 if maybe_email.endswith('.edu') else 0
		elif i == 8:
			feature_vector[i] = 1 if maybe_email.endswith('.tw') else 0
		elif i == 9:
			feature_vector[i] = 1 if len(maybe_email) > 10 else 0
	return feature_vector


def read_in_data(filename):
	"""
	:return: list, containing strings that might be valid email addresses
	"""
	# TODO:
	valid_email = []
	with open(DATA_FILE, 'r') as f:
		em = f.read()
		email = em.split('\n')
	for line in email:
		valid_email.append(line)
	return valid_email



if __name__ == '__main__':
	main()
