import numpy as np 
import matplotlib.pyplot as plt 

import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader

import pickle 
import os 

from lstm_model import * 
from preprocess import * 
from notes_dataset import * 

EPOCHS = 100
BATCH_SIZE = 128
MODEL_PATH = 'LSTMModel/best_model.pth'

print('EPOCHS: {}\nBATCH_SIZE: {}'.format(EPOCHS, BATCH_SIZE))

if torch.cuda.is_available():  
  device = 'cuda:0' 
else:  
  device = 'cpu' 

print('Using device: {}'.format(device))


def train():
    notes, vocab_size, _ = load_data()
    input_sequences, output_sequences = prepare_sequences(notes, vocab_size, 'train')

    model = LSTMModel(input_dim=input_sequences.shape[1:], hidden_dim=512, vocab_size=vocab_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    start_epoch, min_loss = 0, 100 
    loss_values, block_loss = [], []  

    if os.path.exists('LSTMModel/best_model.pth'):
        start_epoch, model, optimizer, min_loss = load_checkpoint(MODEL_PATH, model, optimizer)
        print('Loaded checkpoint. Starting epoch {}'.format(start_epoch))

    model = model.to(device)
    model.train()
    
    training_set = NotesDataset(input_sequences, output_sequences)
    trainloader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    loss_function = nn.CrossEntropyLoss().to(device)

    for epoch in range(start_epoch, EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):

            inputs, labels = inputs.to(device), labels.to(device)
            inputs.requires_grad_(True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs).squeeze()
            loss = loss_function(input=outputs, target=labels.long())
            loss.backward()
            optimizer.step()
            
            block_loss.append(loss.item())

            if i % 200 == 0: 
                print('Epoch: {}\tIteration: {}\tLoss: {}'.format(epoch, i, np.array(block_loss).mean()))
                loss_values.append(loss.item())

                if np.array(block_loss).mean() < min_loss:
                    min_loss = np.array(block_loss).mean() 
                    save_dict = {'epoch': epoch, 
                        'state_dict': model.state_dict(), 
                        'optimizer': optimizer.state_dict(), 
                        'min_loss': min_loss,
                        'loss': loss_values}
                    torch.save(save_dict, MODEL_PATH)
                    print('Saving checkpoint. Best loss: {}'.format(min_loss))
                
                block_loss = []

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

    return epoch, model, optimizer, checkpoint['min_loss'], checkpoint['loss'] if 'loss' in checkpoint else []


def main():
    loss_values = train()
    plt.figure()
    plt.plot(loss_values)
    plt.savefig('LSTMModel/loss.png', dpi=600)
    np.save('loss_values_epoch{}_batchsize{}'.format(EPOCHS, BATCH_SIZE), loss_values)


if __name__ == '__main__':
    main()