from mimetypes import init
from turtle import forward
import torch
import torch.nn as nn
from model import EncoderRNN, DecoderRNN, Seq2Seq

class Dualmodel(nn.Module):
    def __init__(self, data_loader, batch_size, cuda_use) -> None:
        super().__init__()
        self.data_loader = data_loader
        self.decode_classes_dict = data_loader.vocab_dict
        self.decode_classes_list = data_loader.vocab_list
        self.cuda_use = cuda_use
        self.batch_size = batch_size
        self.embed_model = nn.Embedding(data_loader.vocab_len, 128)
        self.encode = EncoderRNN(vocab_size=data_loader.classes_len,
                              embed_model=self.embed_model,
                              embed_size=128,
                              hidden_size=512,
                              input_dropout=0.4,
                              dropout=0.5,
                              layers=2,
                              bidirectional=True)
        self.decode = DecoderRNN(vocab_size=data_loader.vocab_len,
                               embed_model=self.embed_model,
                               hidden_size=1024,
                               embed_size=128,
                               classes_size=data_loader.classes_len,
                               input_dropout=0.4,
                               dropout=0.5,
                               layers=2,
                               bidirectional=True)
        self.model = Seq2Seq(self.encode, self.decode)

    def forward(self, input, input_len, target, target_len):
        
        loss = self.model(input, input_len, target, target_len)
        return loss
