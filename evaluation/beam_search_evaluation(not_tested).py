def int_to_word(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

"""
	*Generate a caption for an image, given a pre-trained model and a tokenizer to map integer back to word
	*Uses simple argmax
"""
def generate_caption(model, tokenizer, image, max_length):
	# Seed the generation process
	in_text = 'startseq'
	# Iterate over the whole length of the sequence
	for _ in range(max_length):
		# Integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# Pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# Predict next word
		# The model will output a prediction, which will be a probability distribution over all words in the vocabulary.
		yhat = model.predict([image,sequence], verbose=0)
		# The output vector representins a probability distribution where maximum probability is the predicted word position
		# Take output class with maximum probability and convert to integer
		yhat = np.argmax(yhat)
		# Map integer back to word
		word = int_to_word(yhat, tokenizer)
		# Stop if we cannot map the word
		if word is None:
			break
		# Append as input for generating the next word
		in_text += ' ' + word
		# Stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text
def generate_caption_beam_search(model, tokenizer, image, max_length, beam_index=3):
	# in_text --> [[idx,prob]] ;prob=0 initially
	in_text = [[tokenizer.texts_to_sequences(['startseq'])[0], 0.0]]
	while len(in_text[0][0]) < max_length:
		tempList = []
		for seq in in_text:
			padded_seq = pad_sequences([seq[0]], maxlen=max_length)
			preds = model.predict([image,padded_seq], verbose=0)
			# Take top (i.e. which have highest probailities) `beam_index` predictions
			top_preds = np.argsort(preds[0])[-beam_index:]
			# Getting the top `beam_index` predictions and 
			for word in top_preds:
				next_seq, prob = seq[0][:], seq[1]
				next_seq.append(word)
				# Update probability
				prob += preds[0][word]
				# Append as input for generating the next word
				tempList.append([next_seq, prob])
		in_text = tempList
		# Sorting according to the probabilities
		in_text = sorted(in_text, reverse=False, key=lambda l: l[1])
		# Take the top words
		in_text = in_text[-beam_index:]
	in_text = in_text[-1][0]
	final_caption_raw = [int_to_word(i,tokenizer) for i in in_text]
	final_caption = []
	for word in final_caption_raw:
		if word=='endseq':
			break
		else:
			final_caption.append(word)
	final_caption.append('endseq')
	return ' '.join(final_caption)

"""
	*Evaluate the model on BLEU Score using argmax predictions
"""

def evaluate_model_beam_search(model, images, captions, tokenizer, max_length, beam_index=3):
	actual, predicted = list(), list()
	for image_id, caption_list in tqdm(captions.items()):
		yhat = generate_caption_beam_search(model, tokenizer, images[image_id], max_length, beam_index=beam_index)
		ground_truth = [caption.split() for caption in caption_list]
		actual.append(ground_truth)
		predicted.append(yhat.split())
	print('BLEU Scores :')
	print('A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0.')
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
