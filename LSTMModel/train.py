import numpy as np 
import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader
import pickle 
# from keras.utils import np_utils
from lstm_model import * 
import os 
import matplotlib.pyplot as plt 

EPOCHS = 100
BATCH_SIZE = 128

MODEL_PATH = 'LSTMModel/best_model.pth'

if torch.cuda.is_available():  
  device = 'cuda:0' 
else:  
  device = 'cpu' 

print('Using device: {}'.format(device))

class NotesDataset(Dataset): 
    
    def __init__(self, in_sequences, out_sequences):
        self.in_sequences = in_sequences 
        self.out_sequences = out_sequences 

    def __len__(self):
        return len(self.in_sequences)

    def __getitem__(self, idx): 
        return self.in_sequences[idx], self.out_sequences[idx]


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    if os.path.exists('Classical-Piano-Composer/data/train_notes_input.npy') and os.path.exists('Classical-Piano-Composer/data/train_notes_output.npy'):
        network_input = np.load('Classical-Piano-Composer/data/train_notes_input.npy')
        network_output = np.load('Classical-Piano-Composer/data/train_notes_output.npy')
        return network_input, network_output

    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    #network_output = np.eye(n_vocab, dtype='uint8')[network_output] #np_utils.to_categorical(network_output) 
    network_output = np.array(network_output)
    np.save('Classical-Piano-Composer/data/train_notes_input', network_input)
    np.save('Classical-Piano-Composer/data/train_notes_output', network_output)

    return (network_input, network_output)


def load_data():
    DATA = 'Classical-Piano-Composer/data/train_notes'
    with open(DATA, 'rb') as f: 
        notes = pickle.load(f)
    vocab_size = len(set(notes))
    return notes, vocab_size


def train():
    notes, vocab_size = load_data()
    input_sequences, output_sequences = prepare_sequences(notes, vocab_size)

    model = LSTMModel(input_dim=input_sequences.shape[1:], hidden_dim=512, vocab_size=vocab_size)
    optimizer = torch.optim.Adam(model.parameters())
    start_epoch, min_loss = 0, 100 

    if os.path.exists('LSTMModel/best_model.pth'):
        start_epoch, model, optimizer, min_loss = load_checkpoint(MODEL_PATH, model, optimizer)
        print('Loaded checkpoint. Starting epoch {}'.format(start_epoch))

    model = model.to(device)

    training_set = NotesDataset(input_sequences, output_sequences)
    trainloader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    loss_function = nn.CrossEntropyLoss().to(device)

    loss_values = []  
    for epoch in range(start_epoch, EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):

            inputs, labels = Variable(inputs.to(device), requires_grad=True), Variable(labels.to(device), requires_grad=False)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float()).squeeze()

            loss = loss_function(input=outputs, target=labels.long())
            loss.backward()

            for param in model.parameters():
                print(param.grad.data.sum())
            optimizer.step()
            if i % 100 == 0: 
                print('Epoch: {}\tIteration: {}\tLoss: {}'.format(epoch, i, loss.item()))
                loss_values.append(loss.item())

            if i % 1000 == 0: 
                if loss < min_loss:
                    min_loss = loss 
                    torch.save({'epoch': epoch, 
                    'state_dict': model.state_dict(), 
                    'optimizer': optimizer.state_dict(), 
                    'min_loss': min_loss}, MODEL_PATH)
                    print('Saving checkpoint. Best loss: {}'.format(loss))
            # print statistics
    
    return loss_values
            

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    return epoch, model, optimizer 


def main():
    loss_values = train()
    plt.figure()
    plt.plot(loss_values)
    plt.savefig('LSTMModel/loss.png', dpi=600)

if __name__ == '__main__':
    main()