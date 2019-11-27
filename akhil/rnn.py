import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

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