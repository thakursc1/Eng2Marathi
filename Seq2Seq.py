from tokenizers import BertWordPieceTokenizer
import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


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
        if not isinstance(decoded, list):
            decoded = [decoded]
        # Hacky way to encode SOS: Start of Sentence and End of Sentence
        decoded = ["[SOS] {} [EOS]".format(i) for i in decoded]
        return self.tokenizer.encode_batch(decoded)

    def decode(self, encoded):
        if not isinstance(encoded, list):
            encoded = [encoded]
        return self.tokenizer.decode_batch(encoded)


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


class MachineTranslationDataset(Dataset):
    def __int__(self, data_file, reverse_translation=False):
        self.reverse_translation = reverse_translation

        # Only the first 2 columns contain the text, ignoring the rest
        data = np.loadtxt(data_file, delimiter='\t', encoding="utf-8", dtype=str)[:, :2]
        self.lang1_sentences = data[:, 0].tolist()
        self.lang2_sentences = data[:, 1].tolist()

        self.lang1_tokenizer = Tokenizer(lang="lang1")
        self.lang2_tokenizer = Tokenizer(lang="lang2")
        self.lang1_tokenizer.train_tokenizer(self.lang1_sentences)
        self.lang2_tokenizer.train_tokenizer(self.lang2_sentences)

        self.transformed_data = []

        # Tokenizing all sentences since its a tiny dataset
        for lang1_sentence, lang2_sentence in zip(self.lang1_sentences, self.lang2_sentences):
            if not self.reverse_translation:
                self.transformed_data.append([self.lang1_tokenizer.encode(lang1_sentence),
                                              self.lang2_tokenizer.encode(lang2_sentence)])
            else:
                self.transformed_data.append([self.lang2_tokenizer.encode(lang2_sentence),
                                              self.lang1_tokenizer.encode(lang1_sentence)])
        self.transformed_data = np.array(self.transformed_data)

    def __len__(self):
        return self.transformed_data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        encoded_pairs = self.transformed_data[idx, :]
        lang1, lang2 = torch.from_numpy(encoded_pairs[:, 0]), torch.from_numpy(encoded_pairs[:, 1])
        return lang1, lang2


def train_seq2seq(dataset):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)