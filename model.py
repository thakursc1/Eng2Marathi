from random import random

from torch import nn
import torch.nn.functional as F
import torch
from utils import MAX_LENGTH, SOS_TOKEN, EOS_TOKEN, DEVICE
from utils import MachineTranslationDataset

HIDDEN_SIZE = 64
EMBEDDING_SIZE = 32


class Seq2Seq(nn.Module):
    def __init__(self, hidden_size, embedding_size, encoder_vocab_len, decoder_vocab_len, decoder_sos_token,
                 decoder_eos_token, attention_mask, num_layers=1, bidirectional=False):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(hidden_size, embedding_size, encoder_vocab_len, num_layers,
                               bidirectional)
        self.decoder = Decoder(hidden_size, embedding_size, decoder_vocab_len, decoder_eos_token,
                               decoder_sos_token, num_layers,
                               bidirectional)
        self.attention_mask = attention_mask

    def forward(self, encoder_input):
        encoder_output, encoder_hidden = self.encoder(encoder_input)
        return self.decoder(encoder_output, encoder_hidden, self.attention_mask)


class Encoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_len, num_layers, bidirectional):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_len, embedding_size)
        self.sequence_layer = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional)

    def forward(self, encoder_input):
        embedded = self.embedding(encoder_input.to(torch.long)).unsqueeze(0)
        output, hidden = self.sequence_layer(embedded)  # hidden_state_0 automatically initialized to 0
        return output, hidden


# Bahdanau style additive Attention  style attention mechanism is used in the
class Attention(nn.Module):
    """
    Implements Bahdanau style additive Attention  style attention
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # TODO: Add code for bidirectional later

        self.Wenc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wdec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.va = nn.Linear(hidden_size, 1, bias=False)

        # to store attention scores
        self.attention_weights = None

    def forward(self, encoder_output, decoder_hidden, attention_mask):
        temp_enc = self.Wenc(encoder_output)
        temp_dec = self.Wdec(decoder_hidden).view(1, -1)
        x = F.tanh(temp_dec + temp_enc)
        energies = self.va(x)  # MLP

        # Mask Energies for Padding where attention mask is False
        attention_mask = ~attention_mask.view(1, -1, 1)
        energies.masked_fill_(attention_mask, -float('inf'))
        self.attention_weights = F.softmax(energies, dim=1)

        # The context vector is the weighted sum of the values.
        context = torch.bmm(self.attention_weights.transpose(1, 2), encoder_output)

        return context, self.attention_weights


class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_len, sos_token, eos_token, num_layers=1, bidirectional=False):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_len, hidden_size)
        self.sequence_layer = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.attention = Attention(hidden_size)
        self.attention_combiner = nn.Linear(hidden_size * 2, embedding_size)
        self.decoded_word_output = nn.Linear(self.hidden_size, vocab_len)
        self.sos_token = torch.tensor(sos_token).type(torch.long).to(DEVICE)
        self.eos_token = torch.tensor(eos_token).type(torch.long).to(DEVICE)

    def forward_step(self, encoder_output, decoder_hidden, decoder_input, attention_mask):
        embedded = self.embedding(decoder_input.type(torch.long)).view(1, 1, -1)
        context, attention_weights = self.attention(encoder_output, decoder_hidden[-1], attention_mask)
        x = torch.cat((embedded, context), 2)
        x = self.attention_combiner(x)

        decoder_output = F.relu(x)
        decoder_output, decoder_hidden = self.sequence_layer(decoder_output, decoder_hidden[-1].view(1, 1, -1))

        decoder_output_logits = F.log_softmax(self.decoded_word_output(decoder_output), dim=1)
        _, decoder_output_token_id = decoder_output.topk(1)
        return decoder_output_token_id, decoder_hidden, attention_weights, decoder_output_logits

    def forward(self, encoder_output, encoder_hidden, attention_mask, max_length=MAX_LENGTH, teacher_forcing=0.5,
                target_output=None):
        output_len = target_output.size(0) if target_output else max_length
        decoder_hidden = encoder_hidden[-1]
        decoder_output = self.sos_token

        decoded_output_sequence = []
        attention_weights_sequence = []
        decoder_logits_sequence = []
        # For every step
        use_teacher_forcing = True if random() < teacher_forcing else False
        for i in range(output_len):
            decoder_output, decoder_hidden, attention_weights, decoder_output_logits = self.forward_step(encoder_output,
                                                                                                         decoder_hidden,
                                                                                                         decoder_output,
                                                                                                         attention_mask)
            # Store output for decoder
            decoder_logits_sequence.append(decoder_output_logits)
            decoded_output_sequence.append(decoder_output)
            attention_weights_sequence.append(attention_weights)

            if use_teacher_forcing and target_output:
                decoder_output = target_output[i]  # Teacher forcing
            else:

                if decoder_output.item() == self.eos_token.item():
                    break

        return decoded_output_sequence, attention_weights_sequence, decoder_logits_sequence


if __name__ == "__main__":
    # Testing Model Dimensions
    dataset = MachineTranslationDataset(r"data/mar.txt")
    ENCODER_VOCAB_LEN = dataset.lang1_tokenizer.tokenizer.get_vocab_size()
    DECODER_VOCAB_LEN = dataset.lang2_tokenizer.tokenizer.get_vocab_size()
    DECODER_SOS_TOKEN_ID = dataset.lang2_tokenizer.tokenizer.token_to_id(SOS_TOKEN)
    DECODER_EOS_TOKEN_ID = dataset.lang2_tokenizer.tokenizer.token_to_id(EOS_TOKEN)
    example = dataset[0]
    print(example[0], ENCODER_VOCAB_LEN, DECODER_VOCAB_LEN, DECODER_EOS_TOKEN_ID, DECODER_SOS_TOKEN_ID)
    model = Seq2Seq(HIDDEN_SIZE, EMBEDDING_SIZE, ENCODER_VOCAB_LEN, DECODER_VOCAB_LEN, DECODER_SOS_TOKEN_ID,
                    DECODER_EOS_TOKEN_ID, attention_mask=example[0]["attention_mask"])
    model.to(DEVICE)
    output, attention_weights, logits = model(example[0]["token_ids"])
    print(output, attention_weights)
