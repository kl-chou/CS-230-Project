import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
import pandas as pd
from tqdm import tqdm 

def get_files():
    print("Loading data") 
    print('-' * 80 + '\n')
    metadata = pd.read_csv('data/raw_data/maestro-v2.0.0/maestro-v2.0.0.csv')

    splits = ['train', 'validation', 'test']
    filenames = []
    
    for category in splits:

        curr_set = metadata[metadata['split'] == category]
        filenames.append(curr_set.midi_filename.tolist())
            
    print('Returning metadata')
    print('-' * 80 + '\n')
    
    return {name: files for name, files in zip(splits, filenames)}
    

def get_notes(split, midi_filenames):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for filename in tqdm(midi_filenames, total=len(midi_filenames)):
        midi = converter.parse('data/raw_data/maestro-v2.0.0/' + filename)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('Classical-Piano-Composer/data/{}_notes'.format(split), 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def main():
    files = get_files()

    for _set, filenames in files.items():
        print('Processing {} set'.format(_set))
        get_notes(_set, filenames)
        print('Processed {} files'.format(len(filenames)))
        print('-' * 80 + '\n') 

if __name__ == '__main__':
    main()