'''
Prerequisite files:

1) MSCOCO
    go to:
    http://cocodataset.org/#download

    download and unzip:
    - 2014 Train images
    - 2014 Val images
    - 2014 Train/Val annotations

    have this file structure and directories:
    ./data/COCO/annotations/ --> all .json files
    ./data/COCO/train2014/ --> all train images
    ./data/COCO/val2014/ --> all test (val) images
    ./output/COCO

2) Flickr30k
    go to:
    http://shannon.cs.illinois.edu/DenotationGraph/data/index.html

    download and unzip:
    - Flickr 30k images
    - Publicly Distributable Version of the Flickr 30k Dataset
      (tokenized captions only)

    rename:
    - flickr30k-images --> train
    - results_20130124.token --> captions.tsv

    have this file structure and directories:
    ./data/Flickr/annotations/captions.tsv
    ./data/Flickr/train
    ./output/Flickr

-------------------------------------------------------
HOW TO USE THIS FILE

call vocab_builder.create_vocabulary() from your file

create_vocabulary() takes 2 arguments:
1) MSCOCO_build = 1
2) Flickr_build = 1

Change the 1 to a 0 if you do not what it to build a new vocab
'''

from pycocotools.coco import COCO
from collections import Counter
import nltk
import pickle
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Defining the needed paths and parameters for MSCOCO
parameter_dict_MSCOCO = {
    # MSCOCO dataset: stores annotations and images
    'data_dir': './data/COCO',
    # MSCOCO output: Stores model checkpoints and vocab
    'output_dir': './output/COCO',
    # Path for MSCOCO training captions from 'data_dir'
    'train_ann_path': 'annotations/captions_train2014.json',
    # Path for MSCOCO validation captions from 'data_dir'
    'test_ann_path': 'annotations/captions_val2014.json',
    # Vocabulary file name
    'vocabulary_path': 'vocab.pkl',
    # Directory name for the training images in the MSCOCO dataset
    'train_img_dir': 'train2014',
    # Directory name for the validation images in the MSCOCO dataset
    'test_img_dir': 'val2014',
    # Add word to vocabulary only if it appears at least this many times
    'vocab_threshold': 5}

# Defining the needed paths and parameters for Flickr
parameter_dict_Flickr = {
    # Flickr dataset: stores annotations and images
    'data_dir': './data/Flickr',
    # Flickr output: Stores model checkpoints and vocab
    'output_dir': './output/Flickr',
    # Path for Flickr training captions from 'data_dir'
    'train_ann_path': 'annotations/captions.tsv',
    # Path for Flickr validation captions from 'data_dir'
    # 'val_ann_path': 'annotations/val_annotations.tsv',
    # Path for Flickr test captions from 'data_dir'
    # 'test_ann_path': 'annotations/test_annotations.tsv',
    # Vocabulary file name
    'vocabulary_path': 'vocab.pkl',
    # Directory name for the training images in the Flickr dataset
    'train_img_dir': 'train',
    # Directory name for the validation images in the Flickr dataset
    'test_img_dir': 'train',
    # Add word to vocabulary only if it appears at least this many times
    'vocab_threshold': 5}


class DatasetVocabulary(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.index = 0

    def adding_new_word(self, word):
        # Adding a new word to the vocabulary, if it already doesn't exist
        if word not in self.word_to_index:
            self.word_to_index[word] = self.index
            self.index_to_word[self.index] = word
            self.index += 1

    def __call__(self, word):
        if word not in self.word_to_index:
            # If word does not exist in vocabulary, then return unknown token
            return self.word_to_index['<unk>']
        return self.word_to_index[word]

    def __len__(self):
        # Returns the length of the vocabulary
        return len(self.word_to_index)

    def start_token(self):
        # Returns the start token
        return '<start>'

    def end_token(self):
        # Returns the end token
        return '<end>'


# Function for creating the vocabulary for the MSCOCO dataset
def creating_vocabulary_MSCOCO(json_file, parameter_dict):
    '''
    Function to build the MSCOCO vocab_dataset. Count the number of types a
    word has been used and if >= vocab_threshold, add to vocab_dataset.
    '''

    coco_json = COCO(json_file)  # all captions for MSCOCO
    vocab_word_ids = coco_json.anns.keys()  # create all word ids

    word_count = Counter()
    for index, word_id in enumerate(vocab_word_ids):
        # Converting all the words to lower case and tokenizing them
        captions_tokens = nltk.tokenize.word_tokenize(
            str(coco_json.anns[word_id]['caption']).lower())
        word_count.update(captions_tokens)

    # Only consider words which appear >= vocab_threshold
    vocabulary_words = []
    for vocab_word, vocab_word_count in word_count.items():
        if vocab_word_count >= parameter_dict['vocab_threshold']:
            vocabulary_words.append(vocab_word)

    # add start, end, unkown, and padding key works
    vocabulary_dataset = DatasetVocabulary()
    vocabulary_dataset.adding_new_word('<pad>')
    vocabulary_dataset.adding_new_word('<start>')
    vocabulary_dataset.adding_new_word('<end>')
    vocabulary_dataset.adding_new_word('<unk>')

    for index, vocab_word in enumerate(vocabulary_words):
        vocabulary_dataset.adding_new_word(vocab_word)

    return vocabulary_dataset


# Function for creating the vocabulary for the MSCOCO dataset
def creating_vocabulary_Flickr(json_file, parameter_dict):
    '''
    Function to build the Flickr vocab_dataset.
    '''

    word_count = Counter()
    annotations = pd.read_table(
        json_file, sep='\t',
        header=None, names=['image', 'caption'])

    for i in range(annotations.shape[0]):
        caption = str(annotations['caption'][i])
        # Converting all the words to lower case and tokenizing them
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        word_count.update(tokens)

    # We only consider the words which appear more than a particular threshold
    vocabulary_words = []
    for vocab_word, vocab_word_count in word_count.items():
        if vocab_word_count >= parameter_dict['vocab_threshold']:
            vocabulary_words.append(vocab_word)

    # add start, end, unkown, and padding key works
    vocabulary_dataset = DatasetVocabulary()
    vocabulary_dataset.adding_new_word('<pad>')
    vocabulary_dataset.adding_new_word('<start>')
    vocabulary_dataset.adding_new_word('<end>')
    vocabulary_dataset.adding_new_word('<unk>')

    for index, vocab_word in enumerate(vocabulary_words):
        vocabulary_dataset.adding_new_word(vocab_word)

    return vocabulary_dataset


def create_vocabulary(MSCOCO_build=1, Flickr_build=1):
    if MSCOCO_build == 1:
        parameter_dict = parameter_dict_MSCOCO

        # Path to vocab
        vocabulary_path = os.path.join(
            parameter_dict['output_dir'],
            parameter_dict['vocabulary_path'])

        # Build vocab
        vocabulary = creating_vocabulary_MSCOCO(
            json_file=os.path.join(
                parameter_dict['data_dir'],
                parameter_dict['train_ann_path']),
            parameter_dict=parameter_dict_MSCOCO)

        # Save vocab
        with open(vocabulary_path, 'wb') as f:
            pickle.dump(vocabulary, f)

    if Flickr_build == 1:
        parameter_dict = parameter_dict_Flickr

        # Path to vocab
        vocabulary_path = os.path.join(
            parameter_dict['output_dir'],
            parameter_dict['vocabulary_path'])

        # Build vocab
        vocabulary = creating_vocabulary_Flickr(
            json_file=os.path.join(
                parameter_dict['data_dir'],
                parameter_dict['train_ann_path']),
            parameter_dict=parameter_dict_Flickr)

        # Save vocab
        with open(vocabulary_path, 'wb') as f:
            pickle.dump(vocabulary, f)
