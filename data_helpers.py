import numpy as np
import re
from collections import defaultdict

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def load_data(neg_file, pos_file):
    """
    Load tokenized input, labels and build vocabulary dict
    """

    x = []
    y = []
    vocab = defaultdict(float)
    
    with open(neg_file, "r", encoding = "utf-8") as file:
        for line in file:
            sent = clean_str(line).split()
            words = set(sent)
            for word in words:
                vocab[word] += 1
            x.append(sent)
            y.append([1,0])
            
    with open(pos_file, "r", encoding = "utf-8") as file:
        for line in file:
            sent = clean_str(line).split()
            words = set(sent)
            for word in words:
                vocab[word] += 1
            x.append(sent)
            y.append([0,1])
            
    return x, y, vocab
    
def load_word2vec(w2v_file, vocab):
    """
    Load pretrained word vecs from Google word2vec
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    word_vecs = {}
    with open(w2v_file, "rb") as file:
        header = file.readline()
        vocab_size, embedding_size = map(int, header.split())
        binary_len = np.dtype("float32").itemsize * embedding_size
        for line in range(vocab_size):
            #break
            word = b""
            while True:
                char = file.read(1)
                if char == " ".encode():
                    break
                if char != '\n'.encode():
                    word = word + char
            word = word.decode()
            if word in vocab:
                word_vecs[word] = np.fromstring(file.read(binary_len), dtype = "float32")
            else:
                file.read(binary_len)
            if len(word_vecs) == len(vocab):
                break
        return word_vecs

def add_unknown_words(word_vecs, vocab, embedding_size = 300):
    """
    Add random vecs for word don't appear in pretrained vecs.
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, embedding_size)


def get_pretrained_embedding_filter(w2v, embedding_size = 300):
    """
    Build pretrained embedding filter 
    and word2index matrix for mapping word to index in filter respectivevly
    """
    W = []
    word2index = defaultdict(int)
    W.append(np.zeros([embedding_size]))
    index = 1
    for word in w2v:
        word2index[word] = index
        W.append(w2v[word])
        index += 1
    return word2index, np.asarray(W).astype("float32")

def index_data(x, word2index):
    x_indexed = []
    max_length = max([len(sent) for sent in x])
    for sent in x:
        sent_indexed = [word2index[word] for word in sent]
        while len(sent_indexed) < max_length:
            sent_indexed.append(0)
        x_indexed.append(sent_indexed)
    return x_indexed

def split_data(x, y, devset_percentage):
    """
    Shuffled and split data into training set and developement set.
    """
    shuffled_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = np.asarray(x)[shuffled_indices]
    y_shuffled = np.asarray(y)[shuffled_indices]
    split_index = int(-1 * devset_percentage * len(y))
    x_train, x_dev = x_shuffled[:split_index], x_shuffled[split_index:]
    y_train, y_dev = y_shuffled[:split_index], y_shuffled[split_index:]
    return x_train, y_train, x_dev, y_dev
