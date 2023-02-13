"""
File: validEmailAddress_2.py
Name: Linda
----------------------------
Please construct your own feature vectors
and try to surpass the accuracy achieved by
Jerry's feature vector in validEmailAddress.py.
feature1 (invalid) : length > 64 TODO:
feature2 (invalid) : No '@' in str or multiple '@' outside the quotation marks
feature3 (valid) : Some strings before '@' TODO:
feature4 (valid) : Some strings after '@' TODO:
feature5 (valid) : There is a '.' after '@' TODO:
feature6 (valid) : No white space except in " " and follow \ TODO:
feature7 (valid) : Ends with ".com" TODO:
feature8 (valid) : Ends with ".edu" TODO:
feature9 (valid) : Ends with ".tw" TODO:
feature10 (valid) : Punctuation marks between " " TODO:

Accuracy of your model: 0.88462 TODO:
"""

import numpy as np

WEIGHT_V = np.array([                           # The weight vector selected by you
	[-10],                       # (see assignment handout for more details)
	[-8],
	[-9],
	[-9],
	[2.5],
	[6],
	[-8],
	[-9],
	[0.5],
	[-9]
])

DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed


def main():
	global WEIGHT_V
	scores = []
	valid = 0
	v_lst = []

	# read txt into a list
	maybe_email_list = read_in_data(DATA_FILE)
	for maybe_email in maybe_email_list:
		feature_vector = feature_extractor(maybe_email)
		WEIGHT_V = np.squeeze(WEIGHT_V.T)
		score = WEIGHT_V.dot(np.squeeze(feature_vector))
		scores.append(round(score, 5))
		if maybe_email_list.index(maybe_email) <= 12:
			if score <= 0:
				valid += 1
				v_lst.append(maybe_email_list.index(maybe_email))
		else:
			if score > 0:
				valid += 1
				v_lst.append(maybe_email_list.index(maybe_email))
		score = 0
	accu = valid / 26
	print('Accuracy: ', accu)
	# TODO:



def feature_extractor(maybe_email):
	"""
	:param maybe_email: str, the string to be processed
	:return: list, feature vector with 10 values of 0's or 1's
	"""

	# Initialize n array with 10 rows, 1 column
	feature_vector = np.zeros((10, 1))
	# print(len(feature_vector))

	# len > 64 (invalid)
	if feature_vector[0][0] == 0:
		feature_vector[0][0] = 1 if len(maybe_email) > 64 else 0

	# No '@' in str or multiple '@' outside the quotation marks (invalid)
	if feature_vector[1][0] == 0:
		start = -1
		end = -1
		if '@' in maybe_email:
			times = maybe_email.count('@')
			for ch in maybe_email:
				if ch in '\"':
					start = maybe_email.find('\"')
					end = maybe_email.rfind('\"')
					break
				elif ch in '”':
					start = maybe_email.find('“')
					end = maybe_email.rfind('”')
					break
			target = maybe_email[start + 1:end]
			if times == 1:
				for ch in target:
					if '@' not in ch:
						feature_vector[1][0] = 0
			else:
				if start == -1 or end == -1:
					feature_vector[1][0] = 1
				else:
					for ch in target:
						if ch in '@':
							times -= 1
							if times == 1:
								feature_vector[1][0] = 0
								break
					if times != 1:
						feature_vector[1][0] = 1
		else:
			feature_vector[1][0] = 1

	# No strings before '@'
	if feature_vector[2][0] == 0:
		if feature_vector[1][0] == 0:
			feature_vector[2][0] = 1 if maybe_email.split('@')[0] is '' else 0

	# No strings after '@'
	if feature_vector[3][0] == 0:
		if feature_vector[1][0] == 0:
			feature_vector[3][0] = 1 if maybe_email.split('@')[1] is '' else 0

	# There is a '.' after '@' and not continuous
	if feature_vector[4][0] == 0:
		times = 0
		neigh = 0
		if feature_vector[1][0] == 0:
				target = maybe_email.rfind('@')
				tar_str = maybe_email[target:]
				for ch in tar_str:
					if '.' in ch:
						if neigh == 0:
							times += 1
							neigh += 1
						else:
							feature_vector[4][0] = 0
							break
					else:
						neigh = 0
				if neigh == 0 and times > 0:
					feature_vector[4][0] = 1

	# 空格只能存在於引號中，並且前面要有一個反斜線
	if feature_vector[5][0] == 0:
		start = -1
		end = -1
		if " " not in maybe_email:
			feature_vector[5][0] = 1
		else:
			for ch in maybe_email:
				if ch in '\"':
					start = maybe_email.find('\"')
					end = maybe_email.rfind('\"')
					break
				elif ch in '”':
					start = maybe_email.find('“')
					end = maybe_email.rfind('”')
					break
			target = maybe_email[start + 1:end]
			if start != -1 or end != -1:
				if " " in target:
					num = target.index(" ")
					if target[num - 1] != '\\':
						feature_vector[5][0] = 0
					else:
						feature_vector[5][0] = 1
			else:
				feature_vector[5][0] = 0

	# 引號中的字符串必須是點分隔的，或者是組成域內部分的唯一元素
	if feature_vector[6][0] == 0:
		start = -1
		end = -1
		for ch in maybe_email:
			if ch in '\"':
				start = maybe_email.find('\"')
				end = maybe_email.rfind('\"')
				break
			elif ch in '”':
				start = maybe_email.find('“')
				end = maybe_email.rfind('”')
				break
		target = maybe_email[start + 1:end]
		feature_vector[6][0] = 1 if '.' not in target else 0

	# 連續兩個點
	if feature_vector[7][0] == 0:
		times = 0
		neigh = 0
		start = -1
		end = -1
		for ch in maybe_email:
			if ch in '\"':
				start = maybe_email.find('\"')
				end = maybe_email.rfind('\"')
				break
			elif ch in '”':
				start = maybe_email.find('“')
				end = maybe_email.rfind('”')
				break

		for ch in maybe_email:
			if '.' in ch:
				if neigh == 0:
					times += 1
					neigh = 1
					n1 = maybe_email.index(ch)
					if n1 == 0:
						feature_vector[7][0] = 1
						break
				else:
					if n1 < start or n1 > end:
						feature_vector[7][0] = 1
						break
					else:
						feature_vector[7][0] = 0
			else:
				neigh = 0
		if neigh == 0 and times > 0:
			feature_vector[7][0] = 0
		else:
			feature_vector[7][0] = 1


	# Ends with ".tw"
	if feature_vector[8][0] == 0:
		feature_vector[8][0] = 1 if maybe_email.endswith('.tw') else 0

	# Punctuation marks between " "
	if feature_vector[9][0] == 0:
		start = -1
		end = -1
		for ch in maybe_email:
			if ch in '!#$%&\'\\*+-/=?^_`{|}~':
				for ch in maybe_email:
					if ch in '\"':
						start = maybe_email.find('\"')
						end = maybe_email.rfind('\"')
						break
					elif ch in '”':
						start = maybe_email.find('“')
						end = maybe_email.rfind('”')
						break
				target1 = maybe_email[:start]
				target2 = maybe_email[end+1:]
				if start != -1 or end != -1:
					if '!#$%&\'\\*+-/=?^_`{|}~' in target1:
						feature_vector[9][0] = 1
					elif '!#$%&\'\\*+-/=?^_`{|}~' in target2:
						feature_vector[9][0] = 1
					else:
						feature_vector[9][0] = 0
				else:
					feature_vector[9][0] = 1
	return feature_vector


def read_in_data(filename):
	"""
	:return: list, containing strings that might be valid email addresses
	"""
	# TODO:
	valid_email = []
	with open(filename, 'r') as f:
		em = f.read()
		email = em.split('\n')
	for line in email:
		valid_email.append(line)
	return valid_email



if __name__ == '__main__':
	main()
