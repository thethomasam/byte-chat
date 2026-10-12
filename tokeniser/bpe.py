"""
    train:   rustbpe   -- learns the merges from our dataset, in parallel
    use:     tiktoken  -- takes those merges and encodes/decodes fast

    uv run python tokeniser/bpe.py

    from tokeniser.bpe import ByteTokeniser
    tok = ByteTokeniser.load_or_train()
"""
import os
import pickle

import rustbpe
import tiktoken
from datasets import load_dataset

DATASET = "AnishJoshi/nl2bash-custom"
VOCAB_SIZE = 4096
SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer.pkl")

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


class ByteTokeniser:
    """Trained BPE tokeniser: rustbpe learns the merges, tiktoken runs them.

        tok = ByteTokeniser.load_or_train()
        ids = tok.encode("ls -la")
        tok.decode(ids)
    """

    def __init__(self, encoding):
        self.encoding = encoding
        self.bos_id = encoding.encode_single_token("<|bos|>")
        self.end_id = encoding.encode_single_token("<|end|>")

    @classmethod
    def train(cls, text_iterator, vocab_size=VOCAB_SIZE):
        """Learn merges in Rust, then hand them to tiktoken for fast encoding.
        Vocab = 256 bytes + learned merges + special tokens."""
        trainer = rustbpe.Tokenizer()
        trainer.train_from_iterator(
            text_iterator, vocab_size - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN
        )
        mergeable_ranks = {bytes(piece): rank
                           for piece, rank in trainer.get_mergeable_ranks()}
        special = {token: len(mergeable_ranks) + index
                   for index, token in enumerate(SPECIAL_TOKENS)}
        return cls(tiktoken.Encoding(
            name="byte-chat",
            pat_str=trainer.get_pattern(),
            mergeable_ranks=mergeable_ranks,
            special_tokens=special,
        ))

    @classmethod
    def load(cls, path=SAVE_PATH):
        with open(path, "rb") as f:
            return cls(tiktoken.Encoding(name="byte-chat", **pickle.load(f)))

    @classmethod
    def load_or_train(cls, path=SAVE_PATH):
        """Load the saved tokeniser; train on the dataset's train split if missing."""
        if os.path.exists(path):
            return cls.load(path)
        tokenizer = cls.train(iter_text(load_dataset(DATASET, split="train")))
        tokenizer.save(path)
        return tokenizer

    def save(self, path=SAVE_PATH):
        with open(path, "wb") as f:
            pickle.dump({"pat_str": self.encoding._pat_str,
                         "mergeable_ranks": self.encoding._mergeable_ranks,
                         "special_tokens": self.encoding._special_tokens}, f)

    @property
    def vocab_size(self):
        return self.encoding.n_vocab

    def encode(self, text):
        """Plain text -> ids. Never produces special tokens."""
        return self.encoding.encode_ordinary(text)

    def encode_batch(self, texts):
        return self.encoding.encode_ordinary_batch(texts)

    def decode(self, ids):
        return self.encoding.decode(ids)

    def encode_chat(self, messages):
        """Turn a conversation into one flat list of ids:
        <|bos|> <|system|> ... <|end|> <|user|> ... <|end|> <|assistant|> ... <|end|>
        """
        ids = [self.bos_id]
        for message in messages:
            ids.append(self.encoding.encode_single_token(f"<|{message['role']}|>"))
            ids.extend(self.encode(message["content"]))
            ids.append(self.end_id)
        return ids


def row_to_messages(row):
    """One nl2bash row -> chat messages for encode_chat."""
    return [{"role": "user", "content": row["nl_command"]},
            {"role": "assistant", "content": row["bash_code"]}]


def main():
    splits = load_dataset(DATASET)          # train / validation / test
    tokenizer = ByteTokeniser.load_or_train()
    print(f"vocab size: {tokenizer.vocab_size}")

    sample = splits["test"][0]["bash_code"]   # held-out text the tokeniser never saw
    ids = tokenizer.encode(sample)
    print(f"\n   text    {sample!r}")
    print(f"   pieces  {[tokenizer.decode([i]) for i in ids]}")
    print(f"   numbers {ids}")
    print(f"   {len(sample)} letters -> {len(ids)} pieces")
    assert tokenizer.decode(ids) == sample

    ids = tokenizer.encode_chat(row_to_messages(splits["test"][0]))
    print(f"\n   chat    {len(ids)} tokens: {tokenizer.decode(ids)!r}")

    texts = list(iter_text(splits["train"]))
    letters = sum(len(t) for t in texts)
    tokens = sum(len(i) for i in tokenizer.encode_batch(texts))
    print(f"\n   whole dataset: {letters:,} letters -> {tokens:,} tokens "
          f"({letters / tokens:.2f}x compression)")


if __name__ == "__main__":
    main()
