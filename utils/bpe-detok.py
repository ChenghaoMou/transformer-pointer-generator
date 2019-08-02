import argparse
import sentencepiece as spm

from tqdm import tqdm

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='learn bpe like a pro')
	parser.add_argument('--input', '-i', type=str, help='input file')
	parser.add_argument('--output', '-o', type=str, help='Output file')
	args = parser.parse_args()
	with open(args.input, 'r') as input, open(args.output, 'w') as output:
		for line in tqdm(input):
			output.write(''.join(line.split()).replace('_', ' '))
	
