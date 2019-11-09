import numpy as np 
import pytorch as torch 
from pytorch import nn 
import pickle 

MAX_LENGTH = 16
NUM_OFFSETS = 1 
EPOCHS = 100 

def load_data():
    DATA = 'Classical-Piano-Composer/data/train_notes'


def train():
    input_sequences, output_sequences = prepare_sequences(_, _)

    model = LSTMModel(input_dim=input_sequences.shape, hidden_dim=)
    loss = model.forward()


    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = cross_entropy()
            loss.backward()
            optimizer.step()

            # print statistics
            

    print('Finished Training')


def main():
    train()

if __name__ == '__main__':
    main()