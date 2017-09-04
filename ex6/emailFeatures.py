import numpy as np

def emailFeatures(word_indices):
	"""takes in a word_indices vector and
	produces a feature vector from the word indices.
	"""
	vocab_dict = {}
	with open("vocab.txt") as f:
		for line in f:
			(val, key) = line.split()
			vocab_dict[key] = int(val)

	n = len(vocab_dict)

	result = np.zeros((n,1))
	print word_indices
	for idx in word_indices:
		result[int(idx)] = 1
	return result
