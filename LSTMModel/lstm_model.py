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
        print(self.linear1.weights)
        self.relu = nn.ReLU(inplace=False)
        self.linear2 = nn.Linear(in_features=256, out_features=vocab_size)

        #self.output = nn.LogSoftmax()


    def forward(self, measure):
        
        lstm_out1, _ = self.lstm1(measure)
        lstm_out1 = self.dropout(lstm_out1)

        lstm_out2, (h_n, c_n) = self.lstm2(lstm_out1)

        linear_out1 = self.linear1(h_n.permute(1, 0, 2))
        relu_out = self.relu(linear_out1)
        linear_out1 = self.dropout(relu_out)

        linear_out2 = self.linear2(linear_out1)

        return linear_out2


# model.add(LSTM(
#         512,
#         input_shape=(network_input.shape[1], network_input.shape[2]),
#         recurrent_dropout=0.3,
#         return_sequences=True
#     ))
#     model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
#     model.add(LSTM(512))
#     model.add(BatchNorm())
#     model.add(Dropout(0.3))
#     model.add(Dense(256))
#     model.add(Activation('relu'))
#     model.add(BatchNorm())
#     model.add(Dropout(0.3))
#     model.add(Dense(n_vocab))
#     model.add(Activation('softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='rmsprop')