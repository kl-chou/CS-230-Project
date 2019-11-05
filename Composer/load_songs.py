
""" 
load_songs.py: Loads and saves midi files to loadable data. 

Usage:
    load_songs.py --load_path=<path> --save_path=<path>

Options:
    -h --help                               show this screen.
	-l --load_path=<path> 					path to metadata CSV 
	-s --save_path=<path> 					path to directory to hold processed data 
"""
import midi
import os, sys 
import util
import argparse 
import json 
import pandas as pd 
import numpy as np
from docopt import docopt

def load_metadata(filename: str):
	''' 
	@param filepath str: path to CSV file containing the metadata table for the Magenta Maestro dataset. 
	
	@returns d: pd.DataFrame containing the loaded CSV file 
	'''

	d = pd.read_csv(filename)
	print('Loaded metadata for {} midi files.'.format(len(d)))
	return d 
	

def main():
	args = docopt(__doc__)

	patterns = {}
	all_samples = []
	all_lens = []
	print("Loading Songs...") 

	metadata = load_metadata(args['--load_path'])
	data_dir, _ = os.path.split(args['--load_path'])

	splits = ['train', 'validation', 'test']

	for category in splits:
		directory = os.path.join(args['--save_path'], category)
		if not os.path.exists(directory):
			os.makedirs(directory)
		
		curr_set = metadata[metadata['split'] == category]
		print(curr_set.head())
				# if not (path.endswith('.mid') or path.endswith('.midi')):
				# 	continue
				# try:
				# 	samples = midi.midi_to_samples(path)
				# except:
				# 	print "ERROR ", path
				# 	continue
				# if len(samples) < 8:
				# 	continue
					
				# samples, lens = util.generate_add_centered_transpose(samples)
				# all_samples += samples
				# all_lens += lens
		
	# assert(sum(all_lens) == len(all_samples))
	# print('Saving ' + str(len(all_samples)) + ' samples...')
	# all_samples = np.array(all_samples, dtype=np.uint8)
	# all_lens = np.array(all_lens, dtype=np.uint32)
	# np.save('samples.npy', all_samples)
	# np.save('lengths.npy', all_lens)
	# print('Done')

if __name__ == '__main__':
    main()