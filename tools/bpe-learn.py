import argparse
import sentencepiece as spm

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='learn bpe like a pro')
	parser.add_argument('--input', '-i', type=str, help='input file')
	parser.add_argument('--model', '-m', type=str, help='Model prefix')
	parser.add_argument('--vocab_size', '-v', type=int, help='vocab size')
	args = parser.parse_args()

	spm.SentencePieceTrainer.Train(f"--input={args.input} --model_prefix={args.model} --vocab_size={args.vocab_size}")
	
