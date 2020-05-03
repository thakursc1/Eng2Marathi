from model import Seq2Seq, HIDDEN_SIZE, EMBEDDING_SIZE
from utils import MAX_LENGTH, SOS_TOKEN, EOS_TOKEN, DEVICE
from utils import MachineTranslationDataset

from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim
import torch

import os
from tqdm import tqdm


class Trainer:
    def __init__(self, model_dir):
        """
        A Trainer class to make training pipeline easier
        """
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def train(self, num_epochs, model, optimizer, train_iter, validate_iter, target_vocab_size):
        model.train()

        for epoch in range(num_epochs):
            loss = 0
            for idx, example in tqdm(enumerate(train_iter), total = len(train_iter)):
                src, trg = example
                optimizer.zero_grad()
                decoded_output_sequence, attention_weights_sequence, decoder_logits_sequence = \
                    model(src["token_ids"],
                          src["attention_mask"],
                          teacher_forcing=0.5,
                          target_output=trg["token_ids"])
                decoder_logits_sequence = torch.stack(decoder_logits_sequence).to(DEVICE)
                step_loss = F.nll_loss(decoder_logits_sequence.view(-1, target_vocab_size),
                                       trg["token_ids"].view(-1))
                step_loss.backward()
                optimizer.step()
                loss += step_loss.item()
            val_loss = self.evaluate(model, validate_iter, target_vocab_size)
            print("Epoch:{}, Loss:{}, val_loss:{}".format(epoch, loss, val_loss))

    @staticmethod
    def evaluate(model, validate_iter, target_vocab_size):
        model.eval()
        loss = 0
        for idx, example in enumerate(validate_iter):
            src, trg = example
            decoded_output_sequence, attention_weights_sequence, decoder_logits_sequence = \
                model(src["token_ids"],
                      src["attention_mask"])
            decoder_logits_sequence = torch.stack(decoder_logits_sequence).to(DEVICE)
            step_loss = F.nll_loss(decoder_logits_sequence.view(-1, target_vocab_size),
                                   trg["token_ids"].view(-1))

            loss += step_loss.data[0]

        return loss / len(validate_iter)


if __name__ == "__main__":
    # Testing Model Dimensions
    dataset = MachineTranslationDataset(r"data/mar.txt")
    ENCODER_VOCAB_LEN = dataset.lang1_tokenizer.tokenizer.get_vocab_size()
    DECODER_VOCAB_LEN = dataset.lang2_tokenizer.tokenizer.get_vocab_size()
    DECODER_SOS_TOKEN_ID = dataset.lang2_tokenizer.tokenizer.token_to_id(SOS_TOKEN)
    DECODER_EOS_TOKEN_ID = dataset.lang2_tokenizer.tokenizer.token_to_id(EOS_TOKEN)
    model = Seq2Seq(HIDDEN_SIZE, EMBEDDING_SIZE, ENCODER_VOCAB_LEN, DECODER_VOCAB_LEN, DECODER_SOS_TOKEN_ID,
                    DECODER_EOS_TOKEN_ID)
    model.cuda()

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    trainer = Trainer(r"data/models")
    optimizer = optim.Adam(model.parameters())
    print(model)
    trainer.train(100, model, optimizer, train_dataset, val_dataset, DECODER_VOCAB_LEN)
