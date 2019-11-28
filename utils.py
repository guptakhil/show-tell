import torch
import torchvision
import torchvision.transforms as tf
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

import os
import numpy as np
import nltk
from PIL import Image
import time
from collections import Counter
from pycocotools.coco import COCO

class MSCOCO(data.Dataset):
	'''
	Data Loader for MSCOCO dataset.
	'''
	def __init__(self, ann_path, data_path, vocab, data_transform=None):

		self.data_path = data_path # Path for the dataset
		self.vocab = vocab # Vocabulary for the dataset
		self.data_transform = data_transform # Augmentations on the dataset
		self.annotation_ids = list(COCO(ann_path).anns.keys()) # Annotation IDs for MSCOCO dataset
		self.annotation_obj = COCO(ann_path) # Annotation file object

	def __getitem__(self, image_idx):
		'''
		Function to retrieve an image and it's corresponding caption for the dataset.
		'''
		annotation_idx = self.annotation_ids[image_idx] # Retrieving the annotation index corresponding to the image index
		#*print('Annotation id is: ' + str(self.annotation_obj.anns[annotation_idx]))
		image_caption = self.annotation_obj.anns[annotation_idx]['caption'] # Retrieving the caption corresponding to the image index
		#*print('Image Caption is: ' + str(image_caption))
		image_idx = self.annotation_obj.anns[annotation_idx]['image_id'] # Retrieving the image index corresponding to the annotation index
		image_path = self.annotation_obj.loadImgs(image_idx)[0]['file_name'] # Recording the path of the image
		#*print('Image Path is: ' + str(image_path))

		# Applying the data augmentation transformations on the images
		image = Image.open(os.path.join(self.data_path, image_path)).convert('RGB')
		if self.data_transform is not None:
			image = self.data_transform(image)

		# Converting caption to lower case and tokenizing, converting to a tensor for computations
		caption_tokens = nltk.tokenize.word_tokenize(str(image_caption).lower())
		image_target_caption = torch.Tensor([self.vocab('<start>')] + [self.vocab(token) for token in caption_tokens] + [self.vocab('<end>')])

		return image_path, image, image_target_caption

	def __len__(self):
		'''
		Function returning the length of the dataset
		'''
		return len(self.annotation_ids)

def create_batch(data):
	'''
	Function to create batches from images and the corresponding real captions.
	'''

	data.sort(key=lambda x: len(x[2]), reverse=True) # Sorting the data # x[1]
	image_paths, images, captions = zip(*data) 	# Retrieving the images and their corresponding captions	
	images = torch.stack(images, 0) # Stacking the images together
	#*print('Number of captions: ' + str(captions))
	caption_len = [len(caption) for caption in captions] # Writing the lengths of the image captions to a list

	target_captions = torch.zeros(len(captions), max(caption_len)).long()
	#*print('Shape of target_captions: ' + str(target_captions.size()))

	for idx, image_caption in enumerate(captions):
		caption_end = caption_len[idx]
		target_captions[idx, : caption_end] = image_caption[ : caption_end]

	return image_paths, images, target_captions, caption_len

def get_data_loader(vocab, params, run_type):
	'''
	Function to load the required dataset in batches.
	'''

	data_transform = tf.Compose([tf.Resize((224, 224)), 
								tf.RandomHorizontalFlip(),
								tf.RandomVerticalFlip(),
								tf.ToTensor(),
								tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

	if run_type == 'train':
		dataset = MSCOCO(params['ann_path_train'], params['data_path_train'], vocab, data_transform)
		data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=params['batch_size'], 
												shuffle=params['shuffle'], num_workers=params['num_workers'], 
												drop_last=True, collate_fn=create_batch)
	elif run_type == 'test':
		dataset = MSCOCO(params['ann_path_test'], params['data_path_test'], vocab, data_transform) # _Test
		data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=params['batch_size'],
												shuffle=False, num_workers=params['num_workers'],
												collate_fn=create_batch)
	else:
		raise ValueError('Please specify a valid run type for data loader. %s doesn\' exist.'%(run_type))

	return data_loader

def create_caption_word_format(tokenized, vocab, flag_blue=False):
	'''
	Function to convert the tokenized version of sentence to a sentence with words from the vocabulary.
	'''
	caption_words = []
	
	for token in tokenized:
		curr_word = []
		for idx in token:
			if vocab.index_to_word[idx] == vocab.end_token():
				break
			if idx != vocab.word_to_index[vocab.start_token()]:
				curr_word.append(vocab.index_to_word[idx])
		if flag_blue:
			caption_words.append([curr_word])
		else:
			caption_words.append(curr_word)

	return caption_words

def create_checkpoint(cnn, rnn, optimizer, epoch, step, train_loss, params):
	'''
	Function to create a checkpoint for the trained models and their corresponding evaluated metrics.
	'''

	model_file = 'model_' + str(epoch) + '.ckpt'
	torch.save({'encoder_state_dict' : cnn.state_dict(),
				'decoder_state_dict' : rnn.state_dict(),
				'optimizer_state_dict' : optimizer.state_dict(),
				'epoch' : epoch,
				'step' : step}, os.path.join(params['output_dir'], model_file))

	metrics_file = 'model_' + str(epoch) + '_metrics.ckpt'
	torch.save({'train_loss': train_loss}, os.path.join(params['output_dir'], metrics_file))
	print("Checkpoint created for Epoch %d (Step %d)."%(epoch, step))