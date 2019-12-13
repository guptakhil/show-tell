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
import pickle

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from evaluation.evaluation_metrics import evaluate

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
	caption_len = [len(caption) for caption in captions] # Writing the lengths of the image captions to a list

	target_captions = torch.zeros(len(captions), max(caption_len)).long()

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
		dataset = MSCOCO(params['ann_path_test'], params['data_path_test'], vocab, data_transform) # Test
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
	torch.save({
		'encoder_state_dict' : cnn.state_dict(),
		'decoder_state_dict' : rnn.state_dict(),
		'optimizer_state_dict' : optimizer.state_dict(),
		'epoch' : epoch,
		'step' : step
		}, 
		os.path.join(params['output_dir'], model_file))

	metrics_file = 'model_' + str(epoch) + '_metrics.ckpt'
	torch.save({
		'train_loss': train_loss
		}, 
		os.path.join(params['output_dir'], metrics_file))
	print("Checkpoint created for Epoch %d (Step %d)."%(epoch, step))

def test_model(cnn, rnn, optimizer, loss_function, data_loader, vocab, params, model_name_load, device, sub_batch_size=-1, beam_size=0):
	'''
	Function to test the model
	'''
	state_dict = torch.load(os.path.join(params['output_dir'], model_name_load + '.ckpt'), map_location=torch.device(device))
	cnn.load_state_dict(state_dict['encoder_state_dict'])
	rnn.load_state_dict(state_dict['decoder_state_dict'])
	optimizer.load_state_dict(state_dict['optimizer_state_dict'])
	print("Model loaded.")

	test_loss = []
	bleu1_corpus = []
	bleu2_corpus = []
	bleu3_corpus = []
	bleu4_corpus = []
	bleu1 = []
	bleu2 = []
	bleu3 = []
	bleu4 = []
	cider = []
	rouge = []
	target_caption_full = {}
	candidate_caption_full = {}

	if sub_batch_size == -1:
		sub_batch_size = len(data_loader)

	start_time = time.time()
	print('Testing started.')
	print("Sub-batch size - ", sub_batch_size)
	for idx, (img_paths, image, caption, caption_len) in enumerate(data_loader, start = 0):
		if idx == sub_batch_size:
			break
		if device == 'cpu':
			image = Variable(image).cpu()
			caption = Variable(caption).cpu()
		elif device == 'gpu':
			image = Variable(image).cuda()
			caption = Variable(caption).cuda()
		else:
			raise ValueError('Please specify a valid device from ["cpu", "gpu"].')
		target_caption = nn.utils.rnn.pack_padded_sequence(caption, caption_len, batch_first=True)[0]
		cnn_feature = cnn(image)
		rnn_tokenized_sentence = rnn(cnn_feature, caption, caption_len)
		loss = loss_function(rnn_tokenized_sentence, target_caption)
		test_loss.append(loss.data.item())

		rnn_tokenized_sentence_prediction = rnn.sentence_index(cnn_feature, beam_size)
		rnn_tokenized_sentence_prediction = rnn_tokenized_sentence_prediction.cpu().data.numpy()
		predicted_words = create_caption_word_format(rnn_tokenized_sentence_prediction, vocab, False)

		original_sentence_tokenized = caption.cpu().data.numpy()
		target_words = create_caption_word_format(original_sentence_tokenized, vocab, True)

		eval_scores = evaluate(target_words, predicted_words)
		for imgs, tgt, pdt in zip(img_paths, target_words, predicted_words):
			if imgs in target_caption_full.keys():
				target_caption_full[imgs].extend(tgt)
				candidate_caption_full[imgs].extend([pdt])
			else:
				candidate_caption_full[imgs] = []
				target_caption_full[imgs] = tgt
				candidate_caption_full[imgs].append(pdt)

		sf = SmoothingFunction()
		bleu1.append(eval_scores['Bleu_1'])
		bleu2.append(eval_scores['Bleu_2'])
		bleu3.append(eval_scores['Bleu_3'])
		bleu4.append(eval_scores['Bleu_4'])
		cider.append(eval_scores['CIDEr'])
		rouge.append(eval_scores['ROUGE_L'])

		if (idx + 1) % 100 == 0:
			print("Step %d - %0.4f test loss, %0.2f time, %.3f BLEU1, %.3f BLEU2, %.3f BLEU3, %.3f BLEU4, %.3f CIDEr, %.3f ROUGE_L." %(idx + 1, loss, time.time() - start_time, 
					np.mean(bleu1)*100.0, np.mean(bleu2)*100.0, np.mean(bleu3)*100.0, np.mean(bleu4)*100.0, np.mean(cider)*100.0, np.mean(rouge)*100.0))

	print("%0.4f test loss, %0.2f time, %.3f BLEU1, %.3f BLEU2, %.3f BLEU3, %.3f BLEU4, %.3f CIDEr, %.3f ROUGE_L." %(np.mean(test_loss), time.time() - start_time, 
					np.mean(bleu1)*100.0, np.mean(bleu2)*100.0, np.mean(bleu3)*100.0, np.mean(bleu4)*100.0, np.mean(cider)*100.0, np.mean(rouge)*100.0))
	# Save the outputs to file
	with open(os.path.join(params['output_dir'], 'Target_Words_Dict.pickle'), 'wb') as f:
		pickle.dump(target_caption_full, f)

	with open(os.path.join(params['output_dir'], 'Candidate_Words_Dict.pickle'), 'wb') as f:
		pickle.dump(candidate_caption_full, f)

	# ------ Evaluate the BLEU score -------- #
	for img_nm in target_caption_full.keys():
		b1, b2, b3, b4 = 0.0, 0.0, 0.0, 0.0
		for j in range(len(candidate_caption_full[img_nm])):
			b1 += corpus_bleu([target_caption_full[img_nm]] , [candidate_caption_full[img_nm][j]], weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=sf.method4)
			b2 += corpus_bleu([target_caption_full[img_nm]] , [candidate_caption_full[img_nm][j]], weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=sf.method4)
			b3 += corpus_bleu([target_caption_full[img_nm]] , [candidate_caption_full[img_nm][j]], weights=(0.34, 0.33, 0.33, 0.0), smoothing_function=sf.method4)
			b4 += corpus_bleu([target_caption_full[img_nm]] , [candidate_caption_full[img_nm][j]], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf.method4)
		bleu1_corpus.append(b1/len(candidate_caption_full[img_nm]))
		bleu2_corpus.append(b2/len(candidate_caption_full[img_nm]))
		bleu3_corpus.append(b3/len(candidate_caption_full[img_nm]))
		bleu4_corpus.append(b4/len(candidate_caption_full[img_nm]))

	print("%0.4f test loss, %0.2f time, %.3f Final BLEU1, %.3f Final BLEU2, %.3f Final BLEU3, %.3f Final BLEU4" % (np.mean(test_loss), time.time() - start_time, 
					np.mean(np.array(bleu1_corpus))*100.0, np.mean(np.array(bleu2_corpus))*100.0, np.mean(np.array(bleu3_corpus))*100.0, np.mean(np.array(bleu4_corpus))*100.0))
	print('Testing completed.')
