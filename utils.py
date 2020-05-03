from tokenizers import BertWordPieceTokenizer
import os
import numpy as np

import torch
from torch.utils.data import Dataset

DEVICE = "cpu"

if torch.cuda.is_available():
    DEVICE = "cuda"

# Based on the max token length of encoded sentences
MAX_LENGTH = 60
SOS_TOKEN = "[CLS]"
EOS_TOKEN = "[SEP]"


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
            self.tokenizer = BertWordPieceTokenizer(vocab_file=self.vocab)
        else:
            self.tokenizer = BertWordPieceTokenizer()

        self.tokenizer.enable_padding(max_length=MAX_LENGTH)
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)

    def train_tokenizer(self, sentences):
        """
        Train a tokenizer with a list of sentences
        """

        if not os.path.exists(self.vocab):
            print("Training tokenizer for {}".format(self.tokenizer_dir))
            # Hugging Face only accepts a Temp File with sentences for Training Tokenizer
            with open(self.tokenizer_dir + "/data.txt", "w+", encoding="utf-8") as f:
                [f.write(i + "\n") for i in sentences]
            self.tokenizer.train([self.tokenizer_dir + "/data.txt"])
            self.tokenizer.save(self.tokenizer_dir)
            print("Trained a tokenizer with vocab size {}".format(self.tokenizer.get_vocab_size()))

            # Removing the temp file
            os.remove(self.tokenizer_dir + "/data.txt")

    def encode(self, decoded):
        return self.tokenizer.encode(decoded)

    def decode(self, encoded):
        return self.tokenizer.decode_batch(encoded)


class MachineTranslationDataset(Dataset):
    def __init__(self, data_file, reverse_translation=False):
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

        print("Tokenizing all sentences in the dataset...")
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
        src = {"token_ids": torch.tensor(encoded_pairs[0].ids).type(torch.long).to(DEVICE),
               "attention_mask": torch.tensor(encoded_pairs[0].attention_mask).type(torch.bool).to(DEVICE)}
        target = {"token_ids": torch.tensor(encoded_pairs[0].ids).type(torch.long).to(DEVICE)}
        return src, target


if __name__ == "__main__":
    dataset = MachineTranslationDataset(r"data/mar.txt")
    # Test Tokenizer output
    print(dataset[0])
