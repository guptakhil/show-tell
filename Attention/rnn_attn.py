import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Attention_Net(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, nos_filters, num_hidden_units, attention_dim=512):
        super(Attention_Net, self).__init__()
        self.encoder_att = nn.Linear(nos_filters, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(num_hidden_units, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, img_feat, hidden_state):
        #*print('Shape of img_feat: ' + str(img_feat.size())); print('Shape of hidden_state: ' + str(hidden_state.size()));
        att1 = self.encoder_att(img_feat)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(hidden_state)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        #*print('Shape of alpha: ' + str(alpha.size()))
        attention_weighted_encoding = (img_feat * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        #*print('Shape of attention_weighted_encoding: ' + str(attention_weighted_encoding.size()))

        return attention_weighted_encoding, alpha

class RNN_Attn(nn.Module):

	def __init__(self, embed_dim, nos_filters, attention_dim, num_hidden_units, vocab_size, num_layers):
		'''
		Args:
			embed_dim (int) : Embedding dimension between CNN and RNN
			num_hidden_units (int) : Number of hidden units
			vocab_size (int) : Size of the vocabulary
			num_layers (int) : # of layers
		'''

		super(RNN_Attn, self).__init__()

		self.nos_filters = nos_filters
		self.num_layers = num_layers
		self.vocab_size = vocab_size
		self.embeddings = nn.Embedding(vocab_size, embed_dim)
		self.unit = nn.GRU(2*embed_dim, num_hidden_units, num_layers, batch_first=True) # 
		self.linear = nn.Linear(num_hidden_units, vocab_size)
		# Define the maximum caption length for testing
		self.cap_max_size = 25 
		self.init_h = nn.Linear(nos_filters, num_hidden_units)  # linear layer to find initial hidden state of LSTMCell
		# Define the attention network
		self.attn = Attention_Net(nos_filters, num_hidden_units, attention_dim)
		# Define the Embedding Mapping
		self.embed = nn.Linear(nos_filters, embed_dim)

	def rnn_iterator(self, caption_embedding, caption_size, cnn_feature, is_train=True): # Core module for forward pass
		batch_size = caption_embedding.size()[0]
		hidden = (self.init_h(cnn_feature.mean(dim=2))).unsqueeze(1).repeat(1, self.num_layers, 1) # nn.LeakyReLU(negative_slope=0.2, in_place=False) ()
		if is_train: # In the training regime
			predictions = Variable(torch.zeros(batch_size, caption_embedding.size(1), self.vocab_size)).cuda()
			alphas = Variable(torch.zeros(batch_size, caption_embedding.size(1), cnn_feature.size(2))).cuda()
			for t in range(caption_embedding.size(1)): # Iterate over time
				#*print('At time: ' + str(t))
				batch_size_t = sum([l > t for l in caption_size]) # Number of samples in the batch with caption length > t
				attn_feat, alpha = self.attn(cnn_feature[:batch_size_t, :, :].transpose(1, 2), hidden[:batch_size_t, -1, :]) # Compute attention
				_, hidden = self.unit(torch.cat([caption_embedding[:batch_size_t, t, :], self.embed(attn_feat[:batch_size_t, :])], dim=1).unsqueeze(1), hidden[:batch_size_t, :, :].transpose(0, 1).contiguous())
				op = self.linear(_.squeeze(1))  
				predictions[:batch_size_t, t, :] = op
				alphas[:batch_size_t, t, :] = alpha
				hidden = hidden.transpose(0, 1) # Move batch to the first dimension again

			return predictions, alphas
		else:  # In the testing regime
			predicted_sentence_idx = []
			for t in range(self.cap_max_size): # Iterate over time
				#*print('At word prediction time: ' + str(t))
				attn_feat, __ = self.attn(cnn_feature.transpose(1, 2), hidden[:, -1, :]) # Compute attention
				#*if t == 0:
				_, hidden = self.unit(torch.cat([caption_embedding[:, 0, :], self.embed(attn_feat[:, :])], dim=1).unsqueeze(1), hidden.transpose(0, 1).contiguous())
				#*else:
				#*	_, hidden = self.unit(torch.cat([caption_embedding[:, 0, :], attn_feat[:, :]], dim=1).unsqueeze(1), hidden)

				op = self.linear(_.squeeze(1))
				predicted_tokenized_word = op.max(1)[1]
				predicted_sentence_idx.append(predicted_tokenized_word)
				caption_embedding = self.embeddings(predicted_tokenized_word)
				caption_embedding = caption_embedding.unsqueeze(1)
				hidden = hidden.transpose(0, 1) # Move batch to the first dimension again

			return predicted_sentence_idx

		return 

	def forward(self, cnn_feature, image_caption, caption_size): # Forward routine when cross-entropy loss is desired

		#*print('Shape of Caption Embedding is: ' + str(image_caption.size()))
		caption_embedding = self.embeddings(image_caption)
		#*print('Shape of Caption Embedding is: ' + str(caption_embedding.size()))
		is_train = True #*; print('Caption size is:' + str(caption_size)); print('CNN Feature size is: ' + str(cnn_feature.size()));
		pred_cap, alphas = self.rnn_iterator(caption_embedding, caption_size, cnn_feature, is_train) # 
		#*coupled_embed = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.map_linear(torch.cat(, )))
		#*torch_raw_embeddings = torch.cat((cnn_feature.unsqueeze(1), caption_embedding), 1)
		#*torch_packed_embeddings = nn.utils.rnn.pack_padded_sequence(caption_embedding, caption_size, batch_first=True) # (torch_raw_embeddings, caption_size, batch_first=True)
		#*print('Batch Sizes of Caption Embedding is: ' + str(torch_packed_embeddings.batch_sizes))
		#*print('Shape of Sorted Sequence is: ' + str(torch_packed_embeddings.data.size()))
		#*print('Sorted Sequence is: ' + str(torch_packed_embeddings.sorted_indices))
		#*print(caption_embedding[4, :, 5])
		#*print(torch_packed_embeddings.data[4, :, 5])
		#*torch_packed_embeddings_unit= self.unit(torch_packed_embeddings)[0]
		#*tokenized_predicted_sentence = self.linear(torch_packed_embeddings_unit[0])
		tokenized_predicted_sentence = nn.utils.rnn.pack_padded_sequence(pred_cap, caption_size, batch_first=True)[0]
		#*print('Shape of final output is: ' + str(tokenized_predicted_sentence.size()))

		return tokenized_predicted_sentence, alphas

	def sentence_index(self, cnn_feature, vocab): # Forward routine when word prediction is desired

		#*caption_max_size = 25 
		#*rnn_hidden_state = None
		#*rnn_data = cnn_feature.unsqueeze(1)

		#*predicted_sentence_idx = []
		ind = vocab('<start>') # Initialize sentences with start indices
		image_caption = Variable(torch.LongTensor(np.ones((cnn_feature.size(0), 1))) * ind).cuda()
		caption_embedding = self.embeddings(image_caption) #*; print('Embeddings done')
		is_train = False

		predicted_sentence_idx = self.rnn_iterator(caption_embedding, None, cnn_feature, is_train)

		'''for idx in range(caption_max_size):

			#*next_state, rnn_hidden_state = self.unit(rnn_data, rnn_hidden_state)
			result_state = pred_cap[:, idx, :] # self.linear(next_state.squeeze(1))
			predicted_tokenized_word = result_state.max(1)[1]
			predicted_sentence_idx.append(predicted_tokenized_word)
			rnn_data = self.embeddings(predicted_tokenized_word)
			rnn_data = rnn_data.unsqueeze(1)'''

		predicted_sentence_idx = torch.stack(predicted_sentence_idx, 1).squeeze()

		return predicted_sentence_idx
