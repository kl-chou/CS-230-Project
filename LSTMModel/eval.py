import torch 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from preprocess import * 
from lstm_model import * 
from notes_dataset import * 


if torch.cuda.is_available():  
  device = 'cuda:0' 
else:  
  device = 'cpu' 


def eval_(model, set_name):
    notes, vocab_size, notes_to_int = load_data(set_name)
    input_seq, output_seq = prepare_sequences(notes, vocab_size, set_name, notes_to_int)
    
    training_set = NotesDataset(input_seq, output_seq)
    trainloader = DataLoader(training_set, batch_size=1, shuffle=False, num_workers=4)
    
    predictions = np.array([], dtype=np.int64)
    for i, (inputs, labels) in tqdm(enumerate(trainloader), total=len(output_seq)):

        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs).squeeze()
        outputs = outputs.cpu().detach().numpy()
        pred = np.argmax(outputs)
        predictions = np.hstack((predictions, pred))

    print(predictions[:10])
    print(output_seq[:10])
    accuracy = np.sum(predictions == output_seq) / len(predictions)
    print('{} accuracy: {}'.format(set_name, accuracy))
    np.save('LSTMModel/predictions_{}'.format(set_name), predictions)
        
def load_model(model, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model 


def main():
    model = LSTMModel(input_dim=(100, 1), hidden_dim=512, vocab_size=1447)
    model = load_model(model, 'LSTMModel/best_model.pth') 

    model = model.to(device)
    model.eval()

    sets = ['train', 'validation', 'test']
    for name in sets:
        eval_(model, name)

if __name__ == '__main__':
    main()