# Transformer(XL) with Pointer Generator for Machine Translation

Currently, it is only transformer with pointer generator! 

It supports:

- Dynamic vocab during generation
- Beam search

# TODO


- Transformer-XL
- Code Cleanup and comments

# Simple Copy Task

## Preprocess
Output *random.data* stores base vocabulary and all corpora.

```bash
python preprocess.py --train_src random.src --train_tgt random.tgt --valid_src random.src --valid_tgt random.tgt --save_data random.data --src_vocab random.vocab --tgt_vocab random.vocab
```

## Train

```bash
python train.py --data random.data --model tfm --layers 2 --batch_size 500 --steps 300 --valid_steps 100 --save_steps 100 --device cuda
```

## Translate

```bash
python eval.py --test_src random.src --test_tgt random.tgt --model tfm-300.pt

```

# Result

Base vocabulary includes all letters, numbers are added as UNKs.

- First line is what model sees during inference/training;
- Second line is for used for dynamic vocab during generation;
- Third line is model's output;

```text
   
    <UNK> V E <UNK> w W <UNK> C u x n D c A w v <UNK> L A t r L f o g Z k <UNK> l E
    5 V E 1 w W 2 C u x n D c A w v 7 L A t r L f o g Z k 6 l E
    5 V E 1 w W u x x x n D c A w v 7 L 7 f f f f o g g g 6 E
    
    
    <UNK> M q <UNK> Q k <UNK> T <UNK> n G b <UNK> <UNK> N h u <UNK> B <UNK> n g h b K d t t n z
    3 M q 8 Q k 1 T 7 n G b 7 6 N h u 7 B 2 n g h b K d t t n z
    3 M 8 q Q k 1 7 7 7 G b b 7 N u u B 2 n g g g K b K t t z z
    
    
    x A a n U I J a J J K h o u m m Y <UNK> B z x S I <UNK> M u <UNK> <UNK> x h
    x A a n U I J a J J K h o u m m Y 6 B z x S I 0 M u 5 8 x h
    x A a n I I J J J J K h o u m m Y 6 z x I I 0 0 M M u 8 h

```
