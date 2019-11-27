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
import random
import pickle
import nltk
from itertools import takewhile
from PIL import Image
from datetime import datetime
import time
from collections import Counter
from pycocotools.coco import COCO
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import argparse
import json

from vocab_builder import get_vocabulary

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

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
		image_caption = self.annotation_obj.anns[annotation_idx]['caption'] # Retrieving the caption corresponding to the image index
		image_idx = self.annotation_obj.anns[annotation_idx]['image_id'] # Retrieving the image index corresponding to the annotation index
		image_path = self.annotation_obj.loadImgs(image_idx)[0]['file_name'] # Recording the path of the image

		# Applying the data augmentation transformations on the images
		image = Image.open(os.path.join(self.data_path, image_path)).convert('RGB')
		if self.data_transform is not None:
			image = self.data_transform(image)

		# Converting caption to lower case and tokenizing, converting to a tensor for computations
		caption_tokens = nltk.tokenize.word_tokenize(str(image_caption).lower())
		image_target_caption = torch.Tensor([self.vocab('<start>')] + [self.vocab(token) for token in caption_tokens] + [self.vocab('<end>')])

		return image, image_target_caption

	def __len__(self):
		'''
		Function returning the length of the dataset
		'''
		return len(self.annotation_ids)

def create_batch(data):
	'''
	Function to create batches from images and the corresponding real captions.
	'''

	data.sort(key=lambda x: len(x[1]), reverse=True) # Sorting the data
	images, captions = zip(*data) 	# Retrieving the images and their corresponding captions	
	images = torch.stack(images, 0) # Stacking the images together
	caption_len = [len(caption) for caption in captions] # Writing the lengths of the image captions to a list

	target_captions = torch.zeros(len(captions), max(caption_len)).long()

	for idx, image_caption in enumerate(captions):
		caption_end = caption_len[idx]
		target_captions[idx, : caption_end] = image_caption[ : caption_end]

	return images, target_captions, caption_len

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
		dataset = MSCOCO(params['ann_path_train'], params['data_path_test'], vocab, data_transform)
		data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=params['batch_size'], 
												shuffle=params['shuffle'], num_workers=params['num_workers'], 
												drop_last=True, collate_fn=create_batch)
	elif run_type == 'test':
		dataset = MSCOCO(params['ann_path_test'], params['data_path_test'], vocab)
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

	start_word = [vocab.word_to_index[word] for word in [vocab.start_token()]]
	end_word = lambda idx: vocab.index_to_word[idx] != vocab.end_token()

	caption_words = []
	for idx in takewhile(end_word, tokenized):
		if idx not in start_word:
			caption_words.append(vocab.index_to_word[idx])

	if flag_blue:
		return [[caption_words]]
	else:
		return [caption_words]

class ResNet(nn.Module):
	'''
	Encoding via pretrained ResNet.
	'''

	def __init__(self, resnet_version=101, embed_dim=256):
		'''
		Args:
			embed_dim (int) : Embedding dimension between CNN and RNN
		'''

		super(ResNet, self).__init__()

		if resnet_version == 18:
			resnet_model = models.resnet18(pretrained=True)
		elif resnet_version == 34:
			resnet_model = models.resnet34(pretrained=True)
		elif resnet_version == 50:
			resnet_model = models.resnet50(pretrained=True)
		elif resnet_version == 101:
			resnet_model = models.resnet101(pretrained=True)
		elif resnet_version == 152:
			resnet_model = models.resnet152(pretrained=True)
		else:
			raise ValueError('Please specify a valid ResNet version. %d doesn\'t exist.'%(resnet_version))
		self.model = nn.Sequential(*list(resnet_model.children())[:-1])

		#Training only the last 2 layers for the Resnet model i.e. linear and batchnorm layer
		self.linear1 = nn.Linear(resnet_model.fc.in_features, embed_dim)
		self.bn1 = nn.BatchNorm1d(embed_dim, momentum=0.01)

		#Initializing the weights using normal distribution
		self.linear1.weight.data.normal_(0, 0.05)
		self.bn1.bias.data.fill_(0)

	def forward(self, x):

		x = self.model(x)
		x = Variable(x.data) # Converting to a PyTorch variable
		x = x.view(x.size(0), -1) # Flattening the output of the CNN model
		x = self.bn1(self.linear1(x)) # Applying the linear layer

		return x

class RNN(torch.nn.Module):

	def __init__(self, embed_dim, num_hidden_units, vocab_size, num_layers):
		'''
		Args:
			embed_dim (int) : Embedding dimension between CNN and RNN
			num_hidden_units (int) : Number of hidden units
			vocab_size (int) : Size of the vocabulary
			num_layers (int) : # of layers
		'''

		super(RNN, self).__init__()

		self.embeddings = nn.Embedding(vocab_size, embed_dim)
		self.unit = nn.GRU(embed_dim, num_hidden_units, num_layers, batch_first=True)
		self.linear = nn.Linear(num_hidden_units, vocab_size)

	def forward(self, cnn_feature, image_caption, caption_size):

		caption_embedding = self.embeddings(image_caption)
		torch_raw_embeddings = torch.cat((cnn_feature.unsqueeze(1), caption_embedding), 1)
		torch_packed_embeddings = nn.utils.rnn.pack_padded_sequence(torch_raw_embeddings, caption_size, batch_first=True)
		torch_packed_embeddings_unit= self.unit(torch_packed_embeddings)[0]
		tokenized_predicted_sentence = self.linear(torch_packed_embeddings_unit[0])

		return tokenized_predicted_sentence

	def sentence_index(self, cnn_feature):

		caption_max_size = 25 
		rnn_hidden_state = None
		rnn_data = cnn_feature.unsqueeze(1)

		predicted_sentence_idx = []

		for idx in range(caption_max_size):

			next_state, rnn_hidden_state = self.unit(rnn_data, rnn_hidden_state)
			result_state = self.linear(next_state.squeeze(1))
			predicted_tokenized_word = result_state.max(1)[1]
			predicted_sentence_idx.append(predicted_tokenized_word)
			rnn_data = self.embeddings(predicted_tokenized_word)
			rnn_data = rnn_data.unsqueeze(1)

		predicted_sentence_idx = torch.stack(predicted_sentence_idx, 1).squeeze()

		return predicted_sentence_idx

def create_checkpoint(cnn, rnn, optimizer, epoch, step, train_loss, params):
	'''
	Function to create a checkpoint for the trained models and their corresponding evaluated metrics.
	'''

	model_file = 'model_' + str(epoch) + '.ckpt'
	torch.save({'cnn_state_dict' : cnn.state_dict(),
				'rnn_state_dict' : rnn.state_dict(),
				'optimizer_state_dict' : optimizer.state_dict(),
				'epoch' : epoch,
				'step' : step}, os.path.join(params['output_dir'], model_file))

	metrics_file = 'model_' + str(epoch) + '_metrics.ckpt'
	torch.save({'train_loss': train_loss}, os.path.join(params['output_dir'], metrics_file))
	print("Checkpoint created for Epoch %d (Step %d)."%(epoch, step))

def train_model(data_loader, optimizer, cnn, rnn, loss_function, params, resume_training):
	'''
	Trains the model.
	'''
	if resume_training:
		cnn.load_state_dict(params.state_dict['cnn_state_dict'])
		rnn.load_state_dict(params.state_dict['rnn_state_dict'])
		optimizer.load_state_dict(params.state_dict['optimizer_state_dict'])
		print("Models loaded.")

	cnn.train()
	rnn.train()

	start_time = time.time()

	for epoch in range(params['num_epochs']):

		print("Epoch %d started." %(epoch + 1))

		train_loss = []
		for idx, (image, caption, caption_len) in enumerate(data_loader):
			image = Variable(image).cuda()
			caption = Variable(caption).cuda()
			target_caption = nn.utils.rnn.pack_padded_sequence(caption, caption_len, batch_first=True)[0]
			optimizer.zero_grad()
			cnn_feature = cnn(image)
			rnn_tokenized_sentence = rnn(cnn_feature, caption, caption_len)
			loss = loss_function(rnn_tokenized_sentence, target_caption)
			train_loss.append(loss.data.item())
			loss.backward()
			optimizer.step()
			if (idx + 1) % 5000 == 0:
				create_checkpoint(cnn, rnn, optimizer, epoch + 1, idx + 1, train_loss, params)
			if (idx + 1) % 1 == 0 or (idx + 1) == len(data_loader):
				print("Epoch %d (Step %d) - %0.4f train loss, %0.2f time." %(epoch + 1, idx + 1, loss, time.time() - start_time))

		print("Epoch %d - %0.4f loss, %.2f time. " %(epoch + 1, np.mean(train_loss), time.time() - start))
		create_checkpoint(cnn, rnn, optimizer, epoch + 1, idx + 1, train_loss, params)

def test_model(data_loader, optimizer, cnn, rnn, loss_function, params, vocab, load_model):
	'''
	Test the model.
	'''
	state_dict = torch.load(os.path.join(params['output_dir'], load_model))
	cnn.load_state_dict(state_dict['cnn_state_dict'])
	rnn.load_state_dict(state_dict['rnn_state_dict'])
	optimizer.load_state_dict(state_dict['optimizer_state_dict'])

	cnn.eval()
	rnn.eval()

	test_loss = []
	bleu1 = []
	bleu4 = []

	for idx, (image, caption, caption_len) in enumerate(data_loader, start = 0):
		image = Variable(image).cuda()
		caption = Variable(caption).cuda()
		target_caption = nn.utils.rnn.pack_padded_sequence(caption, caption_len, batch_first=True)[0]
		cnn_feature = cnn(image)
		rnn_tokenized_sentence = rnn(cnn_feature, caption, caption_len)
		loss = loss_function(rnn_tokenized_sentence, target_caption)
		test_loss.append(loss.data.item())

		rnn_tokenized_sentence_prediction = rnn.sentence_index(cnn_feature)
		rnn_tokenized_sentence_prediction = rnn_tokenized_sentence_prediction.cpu().data.numpy()
		predicted_words = create_caption_word_format(rnn_tokenized_sentence_prediction, vocab, False)

		original_sentence_tokenized = caption.cpu().data.numpy()
		target_words = create_caption_word_format(original_sentence_tokenized, vocab, True)

		sf = SmoothingFunction()
		bleu1.append(corpus_bleu(target_words, predicted_words, weights=(1, 0, 0, 0), smoothing_function=sf.method4))
		bleu4.append(corpus_bleu(target_words, predicted_words, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf.method4))

		if (idx + 1) % 1 == 0:
			print("Epoch %d (Step %d) - %0.4f test loss, %0.2f time, %.3f BLEU1, %.3f BLUE4." %(epoch + 1, idx + 1, loss, time.time() - start_time))

	print("Epoch %d - %0.4f loss, %.2f time,  %.3f BLEU1, %.3f BLUE4." %(epoch + 1, np.mean(test_loss), time.time() - start),
																		np.mean(bleu1), np.mean(bleu4))

def main():

	data_source = 'MSCOCO'

	with open('config.json') as json_file:
		config = json.load(json_file)
	config = config[data_source]

	parser = argparse.ArgumentParser()

	parser.add_argument('--data_dir', type=str, default=config['data_dir'], help="path to the data directory")
	parser.add_argument('--output_dir', type=str, default=config['output_dir'], help="path to the output directory")
	parser.add_argument('--train_ann_path', type=str, default=config['train_ann_path'], help="path to training annotations")
	parser.add_argument('--test_ann_path', type=str, default=config['test_ann_path'], help="path to validation annotations")
	parser.add_argument('--vocabulary_path', type=str, default=config['vocabulary_path'], help="path to the vocabulary file")
	parser.add_argument('--train_img_dir', type=str, default=config['train_img_dir'], help="path to the images for training")
	parser.add_argument('--test_img_dir', type=str, default=config['test_img_dir'], help="path to the images for validation")

	parser.add_argument('--vocab_threshold', type=int, default=config['vocab_threshold'], help="threshold for including words in the data vocabulary")
	parser.add_argument('--embedding_length', type=int, default=config['embedding_length'], help="length of the mebedding to be used by CNN and RNN")
	parser.add_argument('--num_hidden_units', type=int, default=config['num_hidden_units'], help="hidden units to be used by the RNN")
	parser.add_argument('--optimizer_type', type=str, default='SGD', help="optimizer to be used at the time of training")
	parser.add_argument('--resnet_version', type=int, default=101, help="ResNet version to be used for the encoding job")
	parser.add_argument('--lr', type=float, default=config['lr'], help="learning rate")
	parser.add_argument('--momentum', type=float, default=config['momentum'], help="momentum for the optimizer")
	parser.add_argument('--num_epochs', type=int, default=config['num_epochs'], help="number of epochs for training")
	parser.add_argument('--num_layers', type=int, default=config['num_layers'], help="layers to be used by the RNN")
	parser.add_argument('--batch_size', type=int, default=config['batch_size'], help="batch size to be used for training data loader")
	parser.add_argument('--shuffle', type=bool, default=config['shuffle'], help="indicator for shuffling the training data while loading")
	parser.add_argument('--num_workers', type=int, default=config['num_workers'], help="num of workers")

	obj = parser.parse_args()
	params = vars(obj)
	print("Parameters being used by the Model - ", params)

	params['vocab_path'] = os.path.join(params['output_dir'], params['vocabulary_path'])
	params['ann_path_train'] = os.path.join(params['data_dir'], params['train_ann_path'])
	params['data_path_train'] = os.path.join(params['data_dir'], params['train_img_dir'])
	params['ann_path_test'] = os.path.join(params['data_dir'], params['test_ann_path'])
	params['data_path_test'] = os.path.join(params['data_dir'], params['test_img_dir'])

	vocab = get_vocabulary(data_source, params)
	print('Vocabulary loaded.')

	train_data_loader = get_data_loader(vocab, params, 'train')
	test_data_loader = get_data_loader(vocab, params, 'test')
	print("Training and testing data loaded.")

	cnn = ResNet(params['resnet_version'], params['embedding_length'])
	rnn = RNN(params['embedding_length'], params['num_hidden_units'], len(vocab), params['num_layers'])
	trainable_params = list(rnn.parameters()) + list(cnn.linear1.parameters()) + list(cnn.bn1.parameters())
	
	loss_fn = nn.CrossEntropyLoss()
	if params['optimizer_type'] == 'SGD':
		optimizer = torch.optim.SGD(trainable_params, lr=params['lr'], momentum=params['momentum'])
	elif params['optimizer_type'] == 'Adam':
		optimizer = torch.optim.Adam(trainable_params, lr=params['lr'])
	else:
		raise ValueError('Please specify a valid optimizer. %s is invalid.'%(params['optimizer_type']))

	cnn.cuda()
	rnn.cuda()
	loss_fn.cuda()
	print('Loaded the models to the GPU.')

	print('Training started.')
	resume_training = False
	if resume_training:
		start_epoch = 1 # Specify the epoch for the model to be loaded
		params['state_dict'] = os.path.join(params['output_dir'], 'model_' + str(start_epoch + 1) + '.ckpt')
	train_model(train_data_loader, optimizer, cnn, rnn, loss_fn, params, resume_training)
	print('Training completed.')

	print('Testing started.')
	test_model(test_data_loader, optimizer, cnn, rnn, loss_fn, params, vocab, 'model_24.ckpt')
	print('Testing completed.')

if __name__ == "__main__":
	main()