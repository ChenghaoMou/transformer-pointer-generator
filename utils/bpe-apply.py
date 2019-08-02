import argparse
import sentencepiece as spm

from tqdm import tqdm

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='Applying BPE')
	parser.add_argument('--input', '-i', type=str, help='input file')
	parser.add_argument('--model', '-m', type=str, help='Model file')
	parser.add_argument('--output', '-o', type=str, help='Output file')
	args = parser.parse_args()
	sp = spm.SentencePieceProcessor()
	sp.Load(args.model)
	with open(args.input, 'r') as input, open(args.output, 'w') as output:
		for line in tqdm(input):
			output.write(' '.join(sp.EncodeAsPieces(line.strip('\r\n '))) + '\n')
	
