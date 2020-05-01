from tokenizers import BertWordPieceTokenizer
import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn


class Tokenizer:
    def __init__(self, lang):
        """
        A Tokenizer class to load and train a custom tokenizer
        Using the Hugging Face tokenization library for the same
        """
        self.tokenizer_dir = r"data/{}".format(lang)
        if not os.path.exists(self.tokenizer_dir):
            os.mkdir(self.tokenizer_dir)
        self.vocab = self.tokenizer_dir + "/vocab.txt"
        if os.path.exists(self.vocab):
            print("Initialized tokenizer using cached vocab file {}".format(self.vocab))
            self.tokenizer = BertWordPieceTokenizer(self.vocab)
        else:
            self.tokenizer = BertWordPieceTokenizer()

    def train_tokenizer(self, sentences):
        """
        Train a tokenizer with a list of sentences
        """
        if not os.path.exists(self.vocab):
            print("Training tokenizer for {}".format(self.tokenizer_dir))
            with open(self.tokenizer_dir + "/data.txt", "w+", encoding="utf-8") as f:
                [f.write(i + "\n") for i in sentences]
            self.tokenizer.train([self.tokenizer_dir + "/data.txt"])
            self.tokenizer.save(self.tokenizer_dir)
            print("Trained a tokenizer with vocab size {}".format(self.tokenizer.get_vocab_size()))

    def encode(self, decoded):
        return self.tokenizer.encode(decoded)

    def decode(self, encoded):
        return self.tokenizer.decode(encoded)


def read_examples(data_file, eng2mar=True):
    data = np.loadtxt(data_file, delimiter='\t', encoding="utf-8", dtype=str)

    english_sentences = data[:, 0].tolist()
    marathi_sentences = data[:, 1].tolist()
    tokenizer_eng = Tokenizer(lang="eng")
    tokenizer_mar = Tokenizer(lang="mar")
    tokenizer_eng.train_tokenizer(english_sentences)
    tokenizer_mar.train_tokenizer(marathi_sentences)
    pairs = []
    for eng, mar in zip(english_sentences, marathi_sentences):
        if eng2mar:
            pairs.append((tokenizer_eng.encode(eng), tokenizer_mar.encode(mar)))
        else:
            pairs.append((tokenizer_mar.encode(mar), tokenizer_eng.encode(eng)))

    return pairs, tokenizer_eng, tokenizer_mar if eng2mar else pairs, tokenizer_mar, tokenizer_eng


class LSTMEncoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_len):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_len, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded)  # hidden_state_0 automatically initialized to 0
        return output, hidden


class LSTMDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_len):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_len, hidden_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.dense = nn.Linear(hidden_size, vocab_len)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        decoded = nn.Softmax()(self.dense(output))
        return output, hidden, decoded


