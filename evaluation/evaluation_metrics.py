# The core part of code was taken from https://github.com/tylin/coco-caption
# and https://github.com/mtanti/coco-caption

import copy
#import re 
import math, os
from collections import defaultdict
import numpy as np
#import pdb
import subprocess
import threading
#import tempfile
#import sys

#import itertools

def bleu_precook(s, n=4):
	'''Count all the n-grams in a sentence.
	Args:
		s: A string caption sentence.
		n: The max length of n-grams.
	Returns:
		Length of the sentence, and a dict mapping n-grams to their counts in the sentence.
	'''
	words = s.split()
	counts = defaultdict(int)
	for k in range(1,n+1):
		for i in range(len(words)-k+1):
			ngram = tuple(words[i:i+k])
			counts[ngram] += 1   
	return (len(words), counts)

def bleu_cook_refs(refs, n=4):
	'''Transform a list of reference sentences into a form usable by bleu_cook_test().
	Args:
		refs: A sequence of string reference sentences.
		n: The max length of n-grams.
	Returns:
		Lengths of the reference sentences, and a dict mapping n-grams to their max counts in the reference sentences.
	'''
	reflen = []
	maxcounts = {}
	for ref in refs:
		precooked = bleu_precook(ref, n)
		rl = precooked[0]
		counts = precooked[1]        
		reflen.append(rl)       
		for (ngram,count) in counts.items():
			maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
	return (reflen, maxcounts)

def bleu_cook_test(test, tup, eff=None, n=4):
	'''Transform a test sentence as a string (together with the cooked reference sentences) into a form usable by BleuScorer.
	Args:
		test: A test sentence.
		tup: Cooked reference sentences from bleu_cook_refs().
		eff: Whether and how to calculate effective reference sentence length. "average", "shortest", or "closest".
		n: The max length of n-grams.
	Returns:
		Result usable by BleuScorer.
	'''
	reflen = tup[0]
	refmaxcounts = tup[1] 
	testlen, counts = bleu_precook(test, n)
	result = {}
	if eff == "closest":
		result["reflen"] = min((abs(l-testlen), l) for l in reflen)[1]
	elif eff == "shortest":
		result["reflen"]  = min(reflen)
	elif eff == "average":
		result["reflen"]  = float(sum(reflen))/len(reflen)
	else:
		result["reflen"] = reflen
	result["testlen"] = testlen
	result["guess"] = [max(0,testlen-k+1) for k in range(1,n+1)]
	result["correct"] = [0]*n
	for (ngram, count) in counts.items():
		result["correct"][len(ngram)-1] += min(refmaxcounts.get(ngram,0), count)
	return result

def cider_precook(s, n=4):
	'''Count all the n-grams in a sentence.
	Args:
		s: A string caption sentence.
		n: The max length of n-grams.
	Returns:
		A dict mapping n-grams to their counts in the sentence.
	'''
	words = s.split()
	counts = defaultdict(int)
	for k in range(1,n+1):
		for i in range(len(words)-k+1):
			ngram = tuple(words[i:i+k])
			counts[ngram] += 1
	return counts

def cider_cook_refs(refs, n=4):
	'''Count all the n-grams in reference sentences.
	Args:
		refs: A sequence of string reference sentences.
		n: The max length of n-grams.
	Returns:
		A list of dicts mapping n-grams to their count in certain reference sentences.
	'''
	return [cider_precook(ref, n) for ref in refs]

def cider_cook_test(test, n=4):
	'''Count all the n-grams in a test sentence.
	Args:
		test: A string test sentence.
		n: The max length of n-grams.
	Returns:
		A dict mapping n-grams to their counts in the test sentence.
	'''
	return cider_precook(test, n)
	
class BleuScorer(object):
	'''Bleu scorer.'''
	
	__slots__ = "n", "crefs", "ctest", "_score", "_ratio", "_testlen", "_reflen", "special_reflen"

	def copy(self):
		''' copy the refs.'''
		new = BleuScorer(n=self.n)
		new.ctest = copy.copy(self.ctest)
		new.crefs = copy.copy(self.crefs)
		new._score = None
		return new

	def __init__(self, test=None, refs=None, n=4, special_reflen=None):
		''' singular instance '''
		self.n = n
		self.crefs = []
		self.ctest = []
		self.cook_append(test, refs)
		self.special_reflen = special_reflen

	def cook_append(self, test, refs):
		'''called by constructor and __iadd__ to avoid creating new instances.'''
		if refs is not None:
			self.crefs.append(bleu_cook_refs(refs))
			if test is not None:
				cooked_test = bleu_cook_test(test, self.crefs[-1])
				self.ctest.append(cooked_test)
			else:
				self.ctest.append(None)
		self._score = None

	def ratio(self, option=None):
		self.compute_score(option=option)
		return self._ratio

	def score_ratio(self, option=None):
		'''return (bleu, len_ratio) pair'''
		return (self.fscore(option=option), self.ratio(option=option))

	def score_ratio_str(self, option=None):
		return "%.4f (%.2f)" % self.score_ratio(option)

	def reflen(self, option=None):
		self.compute_score(option=option)
		return self._reflen

	def testlen(self, option=None):
		self.compute_score(option=option)
		return self._testlen

	def retest(self, new_test):
		if type(new_test) is str:
			new_test = [new_test]
		assert len(new_test) == len(self.crefs), new_test
		self.ctest = []
		for t, rs in zip(new_test, self.crefs):
			self.ctest.append(bleu_cook_test(t, rs))
		self._score = None
		return self

	def rescore(self, new_test):
		''' replace test(s) with new test(s), and returns the new score.'''
		return self.retest(new_test).compute_score()

	def size(self):
		assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
		return len(self.crefs)

	def __iadd__(self, other):
		'''add an instance (e.g., from another sentence).'''
		if type(other) is tuple:
			self.cook_append(other[0], other[1])
		else:
			assert self.compatible(other), "incompatible BLEUs."
			self.ctest.extend(other.ctest)
			self.crefs.extend(other.crefs)
			self._score = None
		return self

	def compatible(self, other):
		return isinstance(other, BleuScorer) and self.n == other.n

	def single_reflen(self, option="average"):
		return self._single_reflen(self.crefs[0][0], option)

	def _single_reflen(self, reflens, option=None, testlen=None):
		if option == "shortest":
			reflen = min(reflens)
		elif option == "average":
			reflen = float(sum(reflens))/len(reflens)
		elif option == "closest":
			reflen = min((abs(l-testlen), l) for l in reflens)[1]
		else:
			assert False, "unsupported reflen option %s" % option
		return reflen

	def recompute_score(self, option=None, verbose=0):
		self._score = None
		return self.compute_score(option, verbose)
		
	def compute_score(self, option=None, verbose=0):
		n = self.n
		small = 1e-9
		tiny = 1e-15 ## so that if guess is 0 still return 0
		bleu_list = [[] for _ in range(n)]

		if self._score is not None:
			return self._score

		if option is None:
			option = "average" if len(self.crefs) == 1 else "closest"

		self._testlen = 0
		self._reflen = 0
		totalcomps = {'testlen':0, 'reflen':0, 'guess':[0]*n, 'correct':[0]*n}

		# for each sentence
		for comps in self.ctest:
			testlen = comps['testlen']
			self._testlen += testlen

			if self.special_reflen is None: ## need computation
				reflen = self._single_reflen(comps['reflen'], option, testlen)
			else:
				reflen = self.special_reflen

			self._reflen += reflen
				
			for key in ['guess','correct']:
				for k in range(n):
					totalcomps[key][k] += comps[key][k]

			# append per image bleu score
			bleu = 1.
			for k in range(n):
				bleu *= (float(comps['correct'][k]) + tiny) \
						/(float(comps['guess'][k]) + small)
				bleu_list[k].append(bleu ** (1./(k+1)))
			ratio = (testlen + tiny) / (reflen + small) ## N.B.: avoid zero division
			if ratio < 1:
				for k in range(n):
					bleu_list[k][-1] *= math.exp(1 - 1/ratio)

			if verbose > 1:
				print(comps, reflen)

		totalcomps['reflen'] = self._reflen
		totalcomps['testlen'] = self._testlen

		bleus = []
		bleu = 1.
		for k in range(n):
			bleu *= float(totalcomps['correct'][k] + tiny) \
					/ (totalcomps['guess'][k] + small)
			bleus.append(bleu ** (1./(k+1)))
		ratio = (self._testlen + tiny) / (self._reflen + small) ## N.B.: avoid zero division
		if ratio < 1:
			for k in range(n):
				bleus[k] *= math.exp(1 - 1/ratio)

		if verbose > 0:
			print (totalcomps)
			print ("ratio:", ratio)

		self._score = bleus
		return self._score, bleu_list


#  Wrapper for BLEU scorer
class Bleu:
	def __init__(self, n=4):
		# default compute Blue score up to 4
		self._n = n
		self._hypo_for_image = {}
		self.ref_for_image = {}

	def compute_score(self, gts, res):

		assert(gts.keys() == res.keys())
		imgIds = gts.keys()

		bleu_scorer = BleuScorer(n=self._n)
		for id in imgIds:
			hypo = res[id]
			ref = gts[id]

			# Sanity check.
			assert(type(hypo) is list)
			assert(len(hypo) == 1)
			assert(type(ref) is list)
			assert(len(ref) >= 1)

			bleu_scorer += (hypo[0], ref)

		#score, scores = bleu_scorer.compute_score(option='shortest')
		score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
		#score, scores = bleu_scorer.compute_score(option='average', verbose=1)

		# return (bleu, bleu_info)
		return score, scores

	def method(self):
		return "Bleu"


class CiderScorer(object):
	'''CIDEr scorer.'''

	def copy(self):
		''' copy the refs.'''
		new = CiderScorer(n=self.n)
		new.ctest = copy.copy(self.ctest)
		new.crefs = copy.copy(self.crefs)
		return new

	def __init__(self, test=None, refs=None, n=4, sigma=6.0):
		''' singular instance '''
		self.n = n
		self.sigma = sigma
		self.crefs = []
		self.ctest = []
		self.document_frequency = defaultdict(float)
		self.cook_append(test, refs)
		self.ref_len = None

	def cook_append(self, test, refs):
		'''called by constructor and __iadd__ to avoid creating new instances.'''
		if refs is not None:
			self.crefs.append(cider_cook_refs(refs))
			if test is not None:
				self.ctest.append(cider_cook_test(test)) ## N.B.: -1
			else:
				self.ctest.append(None) # lens of crefs and ctest have to match

	def size(self):
		assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
		return len(self.crefs)

	def __iadd__(self, other):
		'''add an instance (e.g., from another sentence).'''
		if type(other) is tuple:
			## avoid creating new CiderScorer instances
			self.cook_append(other[0], other[1])
		else:
			self.ctest.extend(other.ctest)
			self.crefs.extend(other.crefs)
		return self
		
	def compute_doc_freq(self):
		'''
		Compute term frequency for reference data.
		This will be used to compute idf (inverse document frequency later)
		The term frequency is stored in the object
		:return: None
		'''
		for refs in self.crefs:
			# refs, k ref captions of one image
			for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
				self.document_frequency[ngram] += 1
			# maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

	def compute_cider(self):
		def counts2vec(cnts):
			'''
			Function maps counts of ngram to vector of tfidf weights.
			The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
			The n-th entry of array denotes length of n-grams.
			:param cnts:
			:return: vec (array of dict), norm (array of float), length (int)
			'''
			vec = [defaultdict(float) for _ in range(self.n)]
			length = 0
			norm = [0.0 for _ in range(self.n)]
			for (ngram,term_freq) in cnts.items():
				# give word count 1 if it doesn't appear in reference corpus
				df = np.log(max(1.0, self.document_frequency[ngram]))
				# ngram index
				n = len(ngram)-1
				# tf (term_freq) * idf (precomputed idf) for n-grams
				vec[n][ngram] = float(term_freq)*(self.ref_len - df)
				# compute norm for the vector.  the norm will be used for computing similarity
				norm[n] += pow(vec[n][ngram], 2)

				if n == 1:
					length += term_freq
			norm = [np.sqrt(n) for n in norm]
			return vec, norm, length

		def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
			'''
			Compute the cosine similarity of two vectors.
			:param vec_hyp: array of dictionary for vector corresponding to hypothesis
			:param vec_ref: array of dictionary for vector corresponding to reference
			:param norm_hyp: array of float for vector corresponding to hypothesis
			:param norm_ref: array of float for vector corresponding to reference
			:param length_hyp: int containing length of hypothesis
			:param length_ref: int containing length of reference
			:return: array of score for each n-grams cosine similarity
			'''
			delta = float(length_hyp - length_ref)
			# measure consine similarity
			val = np.array([0.0 for _ in range(self.n)])
			for n in range(self.n):
				# ngram
				for (ngram,count) in vec_hyp[n].items():
					# vrama91 : added clipping
					val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

				if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
					val[n] /= (norm_hyp[n]*norm_ref[n])

				assert(not math.isnan(val[n]))
				# vrama91: added a length based gaussian penalty
				val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
			return val

		# compute log reference length
		self.ref_len = np.log(float(len(self.crefs)))

		scores = []
		for test, refs in zip(self.ctest, self.crefs):
			# compute vector for test captions
			vec, norm, length = counts2vec(test)
			# compute vector for ref captions
			score = np.array([0.0 for _ in range(self.n)])
			for ref in refs:
				vec_ref, norm_ref, length_ref = counts2vec(ref)
				score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
			# change by vrama91 - mean of ngram scores, instead of sum
			score_avg = np.mean(score)
			# divide by number of references
			score_avg /= len(refs)
			# multiply score by 10
			score_avg *= 10.0
			# append score of an image to the score list
			scores.append(score_avg)
		return scores

	def compute_score(self, option=None, verbose=0):
		# compute idf
		self.compute_doc_freq()
		# assert to check document frequency
		assert(len(self.ctest) >= max(self.document_frequency.values()))
		# compute cider score
		score = self.compute_cider()
		# debug
		# print score
		return np.mean(np.array(score)), np.array(score)

class Cider:
	def __init__(self, test=None, refs=None, n=4, sigma=6.0):
		# set cider to sum over 1 to 4-grams
		self._n = n
		# set the standard deviation parameter for gaussian penalty
		self._sigma = sigma
	
	def compute_score(self, gts, res):
		'''
		Main function to compute CIDEr score
		:param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
				ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
		:return: cider (float) : computed CIDEr score for the corpus
		'''
	
		assert(gts.keys() == res.keys())
		imgIds = gts.keys()
	
		cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)
	
		for id in imgIds:
			hypo = res[id]
			ref = gts[id]
	
			# Sanity check.
			assert(type(hypo) is list)
			assert(len(hypo) == 1)
			assert(type(ref) is list)
			assert(len(ref) > 0)
	
			cider_scorer += (hypo[0], ref)
	
		(score, scores) = cider_scorer.compute_score()
	
		return score, scores
	
	def method(self):
		return "CIDEr"


# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'
# print METEOR_JAR
class Meteor:

	def __init__(self):
		self.env = os.environ
		self.env['LC_ALL'] = 'en_US.UTF_8'
		self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
				'-', '-', '-stdio', '-l', 'en', '-norm']
		self.meteor_p = subprocess.Popen(self.meteor_cmd, \
				cwd=os.path.dirname(os.path.abspath(__file__)), \
				stdin=subprocess.PIPE, \
				stdout=subprocess.PIPE, \
				stderr=subprocess.PIPE,
				env=self.env, universal_newlines=True, bufsize=1)
		# Used to guarantee thread safety
		self.lock = threading.Lock()

	def compute_score(self, gts, res):
		assert(gts.keys() == res.keys())
		imgIds = sorted(list(gts.keys()))
		scores = []

		eval_line = 'EVAL'
		self.lock.acquire()
		for i in imgIds:
			assert(len(res[i]) == 1)
			stat = self._stat(res[i][0], gts[i])
			eval_line += ' ||| {}'.format(stat)

		# Send to METEOR
		self.meteor_p.stdin.write(eval_line + '\n')
		
		# Collect segment scores
		for i in range(len(imgIds)):
			score = float(self.meteor_p.stdout.readline().strip())
			scores.append(score)

		# Final score
		final_score = float(self.meteor_p.stdout.readline().strip())
		self.lock.release()

		return final_score, scores

	def method(self):
		return "METEOR"

	def _stat(self, hypothesis_str, reference_list):
		# SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
		hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
		score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
		self.meteor_p.stdin.write(score_line+'\n')
		return self.meteor_p.stdout.readline().strip()
 
	def __del__(self):
		self.lock.acquire()
		self.meteor_p.stdin.close()
		self.meteor_p.kill()
		self.meteor_p.wait()
		self.lock.release()


def my_lcs(string, sub):
	"""
	Calculates longest common subsequence for a pair of tokenized strings
	:param string : list of str : tokens from a string split using whitespace
	:param sub : list of str : shorter string, also split using whitespace
	:returns: length (list of int): length of the longest common subsequence between the two strings
	Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
	"""
	if(len(string)< len(sub)):
		sub, string = string, sub

	lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

	for j in range(1,len(sub)+1):
		for i in range(1,len(string)+1):
			if(string[i-1] == sub[j-1]):
				lengths[i][j] = lengths[i-1][j-1] + 1
			else:
				lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

	return lengths[len(string)][len(sub)]

class Rouge():
	'''
	Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set
	'''
	def __init__(self):
		# vrama91: updated the value below based on discussion with Hovey
		self.beta = 1.2

	def calc_score(self, candidate, refs):
		"""
		Compute ROUGE-L score given one candidate and references for an image
		:param candidate: str : candidate sentence to be evaluated
		:param refs: list of str : COCO reference sentences for the particular image to be evaluated
		:returns score: int (ROUGE-L score for the candidate evaluated against references)
		"""
		assert(len(candidate)==1)
		assert(len(refs)>0)
		prec = []
		rec = []

		# split into tokens
		token_c = candidate[0].split(" ")
		
		for reference in refs:
			# split into tokens
			token_r = reference.split(" ")
			# compute the longest common subsequence
			lcs = my_lcs(token_r, token_c)
			prec.append(lcs/float(len(token_c)))
			rec.append(lcs/float(len(token_r)))

		prec_max = max(prec)
		rec_max = max(rec)

		if(prec_max!=0 and rec_max !=0):
			score = ((1 + self.beta**2)*prec_max*rec_max)/float(rec_max + self.beta**2*prec_max)
		else:
			score = 0.0
		return score

	def compute_score(self, gts, res):
		"""
		Computes Rouge-L score given a set of reference and candidate sentences for the dataset
		Invoked by evaluate_captions.py
		:param hypo_for_image: dict : candidate / test sentences with "image name" key and "tokenized sentences" as values
		:param ref_for_image: dict : reference MS-COCO sentences with "image name" key and "tokenized sentences" as values
		:returns: average_score: float (mean ROUGE-L score computed by averaging scores for all the images)
		"""
		assert(gts.keys() == res.keys())
		imgIds = gts.keys()

		score = []
		for id in imgIds:
			hypo = res[id]
			ref  = gts[id]

			score.append(self.calc_score(hypo, ref))

			# Sanity check.
			assert(type(hypo) is list)
			assert(len(hypo) == 1)
			assert(type(ref) is list)
			assert(len(ref) > 0)

		average_score = np.mean(np.array(score))
		return average_score, np.array(score)

	def method(self):
		return "Rouge"

def evaluate(target, predicted):
	"""
	Returns the desired scores.

	Args:
		target
		predicted
	"""
	transformed_target = {}
	transformed_predicted = {}
	for i in range(len(target)):
		transformed_target[i] = []
		for target_sentence in target[i]:
			transformed_target[i].append(' '.join(target_sentence))
		transformed_predicted[i] = [' '.join(predicted[i])]

	scorers = [
		(Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
		(Cider(), "CIDEr"),
		(Rouge(), "ROUGE_L"),
		#(Meteor(), "METEOR")
	]

	evalImgs = []
	eval = {}
	imgToEval = {}
	imgIds = transformed_target.keys()

	eval_scores = {}

	for scorer, method in scorers:
		#print ('computing %s score...'%(scorer.method()))
		score, scores = scorer.compute_score(transformed_target, transformed_predicted)
		if type(method) == list:
			for sc, scs, m in zip(score, scores, method):
				#self.setEval(sc, m)
				eval[m] = sc
				#self.setImgToEvalImgs(scs, gts.keys(), m)
				for imgId, score in zip(imgIds, scs):
					if not imgId in imgToEval:
						imgToEval[imgId] = {}
						imgToEval[imgId]["image_id"] = imgId
						imgToEval[imgId][m] = score
				eval_scores[m] = sc
		else:
			#self.setEval(score, method)
			eval[method] = score
			#self.setImgToEvalImgs(scores, gts.keys(), method)
			for imgId, score in zip(imgIds, scores):
				if not imgId in imgToEval:
					imgToEval[imgId] = {}
					imgToEval[imgId]["image_id"] = imgId
					imgToEval[imgId][method] = score
			eval_scores[method] = score

	return eval_scores