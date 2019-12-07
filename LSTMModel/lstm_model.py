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
        self.lstm3 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.bn1 = nn.BatchNorm1d(num_features=input_dim[1], affine=False)
        self.linear1 = nn.Linear(in_features=hidden_dim, out_features=256)
        self.bn2 = nn.BatchNorm1d(num_features=input_dim[1], affine=False)
        self.relu = nn.ReLU(inplace=False)
        self.linear2 = nn.Linear(in_features=256, out_features=vocab_size)

        #self.output = nn.LogSoftmax()


    def forward(self, measure):
        
        lstm_out1, _ = self.lstm1(measure)
        lstm_out1 = self.dropout(lstm_out1)

        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout(lstm_out2)

        lstm_out3, (h_n, c_n) = self.lstm3(lstm_out2)
        h_n = h_n.permute(1, 0, 2)
        h_n = self.bn1(h_n)
        h_n = self.dropout(h_n)

        linear_out1 = self.linear1(h_n)
        
        relu_out = self.relu(linear_out1)
        relu_out = self.bn2(relu_out)
        relu_out = self.dropout(relu_out)

        linear_out2 = self.linear2(relu_out)

        return linear_out2