# Transformer with Pointer Generator for Machine Translation

Currently, it is only transformer with pointer generator!

It supports:

- <del>Dynamic vocab during generation</del>
- Beam search
- Span Positional Encoding
- Decaying Copy Rate

Since BPE almost eliminates the need for OOV copying in Machine Translation, the dev branch implements a new positional encoding and decaying copying rate to guide the model to copy entire span of tokens. If you still need OOV copy capability, use master branch instead.

## Span Positional Encoding

Each token is encoded into one number to show how far it is to the closest space to the left. E.g.

```
▁F ollow ▁him ▁on ▁Twitter ▁at ▁https :// t witter . com / ge or ge na v atr ue .
```

would have

```
0 1 0 0 0 0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
```

As you can see, tokens within the same span are encoded monotonically. It helps the model to recognize the whole span better.

## Decaying Copy Rate

So at step $T$, the model possibly assign a high probability to word like `▁https`, then for the next step, the probability of copying that word would be partially transfered to the next token within the same span. i.e. `://` will have some probability coming from `▁https`.

$$
    p'_t(token_i) = Prob_{t-1}(token_{i-1}) * 0.1 + 0.9 * Prob_t(token_i) \\
    Prob'_t(token) = softmax(p'_t(token))
$$
