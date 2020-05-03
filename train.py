from model import LSTMDecoder, LSTMEncoder, DEVICE
from torch.utils.data import DataLoader

from random import random


class Trainer:
    def __init__(self, encoder, decoder, encoder_optim, decoder_optim, dataset, model_dir, model_save_freq,
                 loss_track_freq, teacher_forcing_prob, device_type=DEVICE):
        """
        A Trainer class to make training pipeline easier
        """
        self.model_dir = model_dir
        self.model_save_freq = model_save_freq
        self.loss_track_freq = loss_track_freq
        self.device_type = device_type
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optim = encoder_optim
        self.decoder_optim = decoder_optim
        self.dataset = dataset

        self.sos_token_encoded = self.dataset.lang2_tokenizer.encode("[SOS]")
        self.eos_token_encoded = self.dataset.lang2_tokenizer.encode("[EOS]")
        self.teacher_forcing_prob = teacher_forcing_prob


    def loss(self):

    def train_iter(self, sentence1, sentence2):

        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()

        encoder_out, encoder_hidden = self.encoder(sentence1)
        decoded_out = self.sos_token_encoded
        decoded_hidden = encoder_hidden

        teacher_forcing = True if random.random() < self.teacher_forcing_prob else False
        # Using teacher forcing algorithm for faster, but unstable convergence of data
        if teacher_forcing:
            for i in range(sentence2.size(0)):
                decoded_out, decoded_hidden = self.decoder(sentence2[i], decoded_hidden)

        else:
            for i in range(sentence2.size(0)):
                decoded_out, decoded_hidden = self.decoder(decoded_out, decoded_hidden)

                if decoded_out = self.eos_token_encoded:
                    break


    def train(self):
        iterator = DataLoader(self.dataset, shuffle=True)
        for epoch, sentence_pair in enumerate(iterator):
            loss = self.train_iter(sentence_pair[0], sentence_pair[1])
