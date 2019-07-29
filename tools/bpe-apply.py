import argparse
import sentencepiece as spm

from tqdm import tqdm

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='learn bpe like a pro')
	parser.add_argument('--input', '-i', type=str, help='input file')
	parser.add_argument('--model', '-m', type=str, help='Model file')
	parser.add_argument('--output', '-o', type=str, help='Output file')
	args = parser.parse_args()
	sp = spm.SentencePieceProcessor()
	sp.Load(args.model)
	with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
		for line in tqdm(fin):
			fout.write(' '.join(sp.EncodeAsPieces(line.strip('\r\n '))) + '\n')
	
