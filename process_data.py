import sys
import pickle
import data_helpers


if __name__ == "__main__":
    w2v_file = sys.argv[1]
    
    print("Loading data ...")
    pos_file, neg_file = "data/rt-polarity.neg", "data/rt-polarity.pos"
    x_tokenized, y, vocab = data_helpers.load_data(pos_file, neg_file)
    print("Data loaded!")
    print("Vocabulary Size: {}".format(len(vocab)))
    print("Number of Samples: {}".format(len(y)))
    
    print("Load word2vec ...")
    w2v = data_helpers.load_word2vec(w2v_file, vocab)
    print("Word2vec loaded!")

    print("Add unknown word...")
    data_helpers.add_unknown_words(w2v, vocab)
    print("Unkown word loaded!")
    
    print("Build pretrained embedding filter...")
    word2index, pretrained_embedding_filter = data_helpers.get_pretrained_embedding_filter(w2v)
    x = data_helpers.index_data(x_tokenized, word2index)
    print("Pretrained embedding filter built!")
    
    pickle.dump([x, y, pretrained_embedding_filter, word2index], open("data.p", "wb"))
