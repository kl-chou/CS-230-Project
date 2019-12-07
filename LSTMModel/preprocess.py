import numpy as np 
import pickle 

def prepare_sequences(notes, n_vocab, set_name, note_to_int=None):
    """ Prepare the sequences used by the Neural Network """
    set_path = 'Classical-Piano-Composer/data/{}_notes'.format(set_name)

    if os.path.exists(set_path + '_input.npy') and os.path.exists(set_path + '_output.npy'):
        network_input = np.load(set_path + '_input.npy')
        network_output = np.load(set_path + '_output.npy')
        
        return network_input, network_output

    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    if set_name == 'train':
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

    np.save(set_path + '_input', network_input)
    np.save(set_path + '_output', network_output)
    if set_name == 'train':
        with open('note_to_int_dict', 'wb') as f:
            pickle.dump(note_to_int, f)


    return network_input, network_output, note_to_int

def load_data():
    DATA = 'Classical-Piano-Composer/data/train_notes'
    with open(DATA, 'rb') as f: 
        notes = pickle.load(f)
    vocab_size = len(set(notes))
    return notes, vocab_size, pickle.load(open('note_to_int_dict'))