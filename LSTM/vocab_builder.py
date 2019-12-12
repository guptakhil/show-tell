from pycocotools.coco import COCO
from collections import Counter
import nltk
import pickle
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

class DatasetVocabulary(object):

	def __init__(self):
		"""
		Initializing the parameters.
		"""
		self.word_to_index = {}
		self.index_to_word = {}
		self.index = 0

	def add_new_word(self, word):
		"""
		Adds a new word to the vocabulary. (if it doesn't already exist)
		"""
		if word not in self.word_to_index:
			self.word_to_index[word] = self.index
			self.index_to_word[self.index] = word
			self.index += 1

	def __call__(self, word):

		if word not in self.word_to_index:
			return self.word_to_index['<unk>']
		else:
			return self.word_to_index[word]

	def __len__(self):
		return len(self.word_to_index)

	def start_token(self):
		return '<start>'

	def end_token(self):
		return '<end>'

def get_vocabulary(dataset, params):
	"""
	Retrieves the vocabulary for the specified dataset.

	Args:
		dataset (str) : 'MSCOCO' or 'Flickr'
		params (dict) : Contains parameters

	Returns:
		vocab_dataset : Dataset having the desired vocabulary
	"""

	if os.path.isfile(params['vocab_path']): # Load from existing file
		with open(params['vocab_path'], 'rb') as f:
			print('Loading vocabulary from the existing file.')
			vocab_dataset = pickle.load(f)

	else: # Create vocabulary from scratch
		print('Vocabulary does not exist. Creating vocab...')

		# Adding necessary keywords to the dataset vocabulary
		vocab_dataset = DatasetVocabulary()
		for word in ['pad', 'start', 'end', 'unk']:
			vocab_dataset.add_new_word('<' + word + '>')

		annotation_path = os.path.join(params['data_dir'], params['train_ann_path'])
		caption_tokens = Counter()

		if dataset == 'MSCOCO':
			print("Building vocabulary for the MSCOCO dataset.")
			coco_json = COCO(annotation_path)

			# Iterating over the captions and converting to lowercase, followed by tokenizing the words
			for word in coco_json.anns.keys():
				caption_tokens.update(nltk.tokenize.word_tokenize(str(coco_json.anns[word]['caption']).lower()))

		elif dataset == 'Flickr':
			print("Building vocabulary for the Flickr dataset.")
			annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])

			# Iterating over the captions and converting to lowercase, followed by tokenizing the words
			for idx in range(annotations.shape[0]):
				caption_tokens.update(nltk.tokenize.word_tokenize(str(annotations['caption'][idx]).lower()))

		else:
			raise ValueError("Please specify a valid dataset. %s is invalid."%(dataset))

		# Including words which exceed the threshold
		for vw, vw_count in caption_tokens.items():
			if vw_count >= params['vocab_threshold']:
				vocab_dataset.add_new_word(vw)

		# Saving the vocabulary
		with open(params['vocab_path'], 'wb') as f:
			pickle.dump(vocab_dataset, f)

	return vocab_dataset