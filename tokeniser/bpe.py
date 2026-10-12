"""
    train:   rustbpe   -- learns the merges from our dataset, in parallel
    use:     tiktoken  -- takes those merges and encodes/decodes fast

    uv run python bpe.py
"""
import os
import pickle

import rustbpe
import tiktoken
from datasets import load_dataset

DATASET = "AnishJoshi/nl2bash-custom"
VOCAB_SIZE = 4096
SAVE_PATH = "tokenizer.pkl"

# GPT-4 style pre-split: words, numbers (max 2 digits), punctuation, whitespace.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = ["<|bos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


def iter_text(dataset):
    """Yield plain strings -- the raw text BPE learns from.
    Instruction and command go in as two separate strings; pairing doesn't
    matter here, BPE only looks at letters inside each string."""
    for row in dataset:
        yield row["nl_command"]
        yield row["bash_code"]


def train(text_iterator, vocab_size=VOCAB_SIZE):
    """Learn merges in Rust, then hand them to tiktoken for fast encoding.

    Returns:
        tiktoken.Encoding -- 256 bytes + learned merges + special tokens.
    """
    trainer = rustbpe.Tokenizer()
    trainer.train_from_iterator(
        text_iterator, vocab_size - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN
    )
    mergeable_ranks = {bytes(piece): rank
                       for piece, rank in trainer.get_mergeable_ranks()}
    special = {token: len(mergeable_ranks) + index
               for index, token in enumerate(SPECIAL_TOKENS)}
    return tiktoken.Encoding(
        name="byte-chat",
        pat_str=trainer.get_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens=special,
    )


def save(encoding, path=SAVE_PATH):
    with open(path, "wb") as f:
        pickle.dump({"pat_str": encoding._pat_str,
                     "mergeable_ranks": encoding._mergeable_ranks,
                     "special_tokens": encoding._special_tokens}, f)


def load(path=SAVE_PATH):
    with open(path, "rb") as f:
        return tiktoken.Encoding(name="byte-chat", **pickle.load(f))


def encode_chat(encoding, messages):
    """Turn a conversation into one flat list of ids:
    <|bos|> <|system|> ... <|end|> <|user|> ... <|end|> <|assistant|> ... <|end|>
    """
    ids = [encoding.encode_single_token("<|bos|>")]
    for message in messages:
        ids.append(encoding.encode_single_token(f"<|{message['role']}|>"))
        ids.extend(encoding.encode_ordinary(message["content"]))
        ids.append(encoding.encode_single_token("<|end|>"))
    return ids


def main():
    splits = load_dataset(DATASET)          # train / validation / test
    dataset = splits["train"]

    if os.path.exists(SAVE_PATH):
        encoding = load()
    else:
        encoding = train(iter_text(dataset))   # learn only from train
        save(encoding)
    print(f"vocab size: {encoding.n_vocab}")

    sample = splits["test"][0]["bash_code"]   # held-out text the tokeniser never saw
    ids = encoding.encode_ordinary(sample)
    print(f"\n   text    {sample!r}")
    print(f"   pieces  {[encoding.decode([i]) for i in ids]}")
    print(f"   numbers {ids}")
    print(f"   {len(sample)} letters -> {len(ids)} pieces")
    assert encoding.decode(ids) == sample

    row = splits["test"][0]
    messages = [{"role": "user", "content": row["nl_command"]},
                {"role": "assistant", "content": row["bash_code"]}]
    ids = encode_chat(encoding, messages)
    print(f"\n   chat    {len(ids)} tokens: {encoding.decode(ids)!r}")

    letters = sum(len(t) for t in iter_text(dataset))
    tokens = sum(len(i) for i in encoding.encode_ordinary_batch(list(iter_text(dataset))))
    print(f"\n   whole dataset: {letters:,} letters -> {tokens:,} tokens "
          f"({letters / tokens:.2f}x compression)")


if __name__ == "__main__":
    main()
