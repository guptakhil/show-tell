import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

import os
import numpy as np
import random
import time
import argparse
import json

from vocab_builder import get_vocabulary
from utils import MSCOCO, get_data_loader, create_caption_word_format, create_checkpoint
from cnn import ResNet
from rnn import RNN

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

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
parser.add_argument('--resume_training', type=bool, default=config['resume_training'], help="indicator for resuming the training")
parser.add_argument('--resume_model_train', type=str, default=config['resume_model_train'], help="model for resuming the training")
parser.add_argument('--is_training', type=int, default=config['is_training'], help="indicates whether the model needs to be trained")
parser.add_argument('--is_testing', type=int, default=config['is_testing'], help="indicates whether the model needs to be tested")
parser.add_argument('--load_model_test', type=str, default=config['load_model_test'], help="model number for inference")

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
print("Training data loaded.")

cnn = ResNet(params['resnet_version'], params['embedding_length'])
rnn = RNN(params['embedding_length'], params['num_hidden_units'], len(vocab), params['num_layers'])
loss_fn = nn.CrossEntropyLoss()

trainable_params = list(rnn.parameters()) + list(cnn.linear_secondlast_layer.parameters()) + list(cnn.last_layer.parameters())
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

if params['is_training']:
	if params['resume_training']:
		print("Loading the model - %s"%(params['resume_model_train'] + '.ckpt'))
		state_dict = torch.load(os.path.join(params['output_dir'], params['resume_model_train'] + '.ckpt'))
		cnn.load_state_dict(state_dict['encoder_state_dict'])
		rnn.load_state_dict(state_dict['decoder_state_dict'])
		optimizer.load_state_dict(state_dict['optimizer_state_dict'])
		print("Models loaded.")

	cnn.train()
	rnn.train()

	start_time = time.time()

	print('Training started.')
	for epoch in range(params['num_epochs']):

		print("Epoch %d started." %(epoch + 1))

		train_loss = []
		for idx, (image, caption, caption_len) in enumerate(train_data_loader):
			image = Variable(image).cuda()
			caption = Variable(caption).cuda()
			target_caption = nn.utils.rnn.pack_padded_sequence(caption, caption_len, batch_first=True)[0]
			optimizer.zero_grad()
			cnn_feature = cnn(image)
			rnn_tokenized_sentence = rnn(cnn_feature, caption, caption_len)
			loss = loss_fn(rnn_tokenized_sentence, target_caption)
			train_loss.append(loss.data.item())
			loss.backward()
			optimizer.step()
			if (idx + 1) % 5000 == 0:
				create_checkpoint(cnn, rnn, optimizer, epoch + 1, idx + 1, train_loss, params)
			if (idx + 1) % 500 == 0 or (idx + 1) == len(train_data_loader):
				print("Epoch %d (Step %d) - %0.4f train loss, %0.2f time." %(epoch + 1, idx + 1, loss, time.time() - start_time))

		print("Epoch %d - %0.4f loss, %.2f time. " %(epoch + 1, np.mean(train_loss), time.time() - start))
		create_checkpoint(cnn, rnn, optimizer, epoch + 1, idx + 1, train_loss, params)

	print('Training completed.')

if params['is_testing']:
	test_data_loader = get_data_loader(vocab, params, 'test')

	state_dict = torch.load(os.path.join(params['output_dir'], params['load_model_test'] + '.ckpt'), map_location=torch.device('cuda'))
	cnn.load_state_dict(state_dict['encoder_state_dict'])
	rnn.load_state_dict(state_dict['decoder_state_dict'])
	optimizer.load_state_dict(state_dict['optimizer_state_dict'])

	cnn.eval()
	rnn.eval()

	test_loss = []
	bleu1 = []
	bleu4 = []

	start_time = time.time()
	print('Testing started.')
	for idx, (image, caption, caption_len) in enumerate(test_data_loader, start = 0):
		image = Variable(image).cuda()
		caption = Variable(caption).cuda()
		target_caption = nn.utils.rnn.pack_padded_sequence(caption, caption_len, batch_first=True)[0]
		cnn_feature = cnn(image)
		rnn_tokenized_sentence = rnn(cnn_feature, caption, caption_len)
		loss = loss_fn(rnn_tokenized_sentence, target_caption)
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
			print("Step %d - %0.4f test loss, %0.2f time, %.3f BLEU1, %.3f BLUE4." %(idx + 1, loss, time.time() - start_time, np.mean(bleu1)*100.0, np.mean(bleu4)*100.0))

	print("%0.4f loss, %.2f time,  %.3f BLEU1, %.3f BLUE4." %(np.mean(test_loss), time.time() - start), np.mean(bleu1)*100.0, np.mean(bleu4)*100.0)
	print('Testing completed.')