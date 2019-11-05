import midi
import os
import util
import numpy as np
import argparse 
import json 
import pandas as pd 


def load_metadata(filepath: str):
	''' 
	@param filepath str: path to CSV file containing the metadata table for the Magenta Maestro dataset. 
	
	@returns d: pd.DataFrame containing the loaded CSV file 
	'''

	d = pd.read_csv(filename)
	print('Loaded metadata for {} midi files.'.format(len(d)))
	return d 
	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('l', '--load_path', help='Filepath to metadata CSV')
	parser.add_argument('s', '--save_path', help='Path to directory holding processed data')
	args = parser.parse_args()

	patterns = {}
	all_samples = []
	all_lens = []
	print("Loading Songs...") 

	metadata = load_metadata(args['--load_path'])
	data_dir, _ = os.path.split(args['--load_path'])
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
		
	assert(sum(all_lens) == len(all_samples))
	print('Saving ' + str(len(all_samples)) + ' samples...')
	all_samples = np.array(all_samples, dtype=np.uint8)
	all_lens = np.array(all_lens, dtype=np.uint32)
	np.save('samples.npy', all_samples)
	np.save('lengths.npy', all_lens)
	print('Done')

if __name__ == '__main__': 
	main() 