# byte-chat
A series of componets of a LLM explained using simple handheld examples.
A single-file, printed walkthrough of **Byte Pair Encoding (BPE)** — the tokenizer
algorithm behind GPT, Claude and Llama — on an example small enough to check by hand.

## Run

```bash
python3 tokeniser.py
```
## What it does

Trains a tiny BPE tokenizer on one sentence (`"the cat sat on the mat the cat ate the rat"`)
and prints every step:

1. split into words, keeping the leading space
2. count the words
3. break them into letters
4. **merge loop** — count neighbouring pairs, glue the most common one, repeat (6 rounds)
5. show the learned vocabulary
6. encode new text with it

Result: 42 letters become 21 pieces, with merges like `'at'`, `'th'`, `'the'`, `' cat'`.

Edit `TEXT` and `ROUNDS` at the top of the file to try your own.
