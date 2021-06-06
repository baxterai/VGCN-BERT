import re

class DataReader():
	"""
	Get dataset from files
	
	Examples:
		train, dev, test = DataReader("data/train.txt","data/dev.txt","data/test.txt").read()
	"""
	def __init__(self, train_file, dev_file, test_file):
		"""
		Init dataset information.

		Inputs:
			train_file: train file's location & full name
			dev_file: dev file's location & full name
			test_file: test file's location & full name

		Examples:
			DataReader("data/train.txt","data/dev.txt","data/test.txt")
		"""
		self.train_file = train_file
		self.dev_file = dev_file
		self.test_file = test_file
		self.train_raw = []
		self.dev_raw = []
		self.test_raw = []
		self.maxSentenceLength = 200	#CHECKTHIS

	def get_raw(self, input_file):
		"""
		Get raw data from file

		Inputs:
			input_file: input file name

		Returns:
			raw_data: a set with raw data

		Examples:
			raw = get_raw("data/train.txt")
		"""
		with open(input_file) as reader:
			raw_data = reader.readlines()
		
		return raw_data

	def formate(self, raw_data):
		"""
		Formate raw data

		Inputs:
			raw_data: a set with raw data

		Returns:
			dataset: a set with formated data

		Examples:
			raw = ["Abc\\n"]
			dataset = raw (split by EOS character '. ']
		"""
		
		dataset = []
		for raw in raw_data:
			paragraphSentences = self.split_sentences(raw)
			
			#ensure max sentence length: 
			for sentence in paragraphSentences:
				words = sentence.split()
				if(len(words) < self.maxSentenceLength):
					dataset.append(sentence)
			
			#dataset.extend(paragraphSentences)

		return dataset

	def read(self):
		"""
		Get dataset and formate.

		Returns:
			train: train dataset
			dev: dev dataset
			test: test dataset

		Examples:
			train, dev, test = read()
		"""
		train = self.formate(self.get_raw(self.train_file))
		dev = self.formate(self.get_raw(self.dev_file))
		test = self.formate(self.get_raw(self.test_file))

		return train, dev, test

	def split_sentences(self, st):
		sentences = re.split(r'[.?!]\s*', st)
		if sentences[-1]:
			return sentences
		else:
			return sentences[:-1]
		
