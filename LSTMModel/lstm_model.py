import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, vocab_size, dropout_prob=0.3):
        super(LSTMModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes measures as inputs, and outputs hidden states
        # with dimensionality hidden_dim.d
        self.lstm1 = nn.LSTM(input_size=input_dim[1], hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.linear1 = nn.Linear(in_features=hidden_dim, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=vocab_size)

        #self.output = nn.LogSoftmax()


    def forward(self, measure):
        
        lstm_out1, _ = self.lstm1(measure)
        lstm_out1 = self.dropout(lstm_out1)

        lstm_out2, _ = self.lstm2(lstm_out1)
        linear_out1 = self.linear1(lstm_out2)
        linear_out1 = self.dropout(linear_out1)

        linear_out2 = self.linear2(linear_out1) # Transform to (vocab_size, 1) output and then softmax to make prediction 
        #logits = self.output(linear_out2)

        return linear_out2