# Transformer with Pointer Generator for Machine Translation

Currently, it is only transformer with pointer generator!

It supports:

- <del>Dynamic vocab during generation</del>
- Beam search

Since BPE almost eliminates the need for OOV copying in Machine Translation, the dev branch implements a new positional encoding and decaying copying rate to guide the model to copy entire span of tokens. If you still need OOV copy capability, use master branch instead.
