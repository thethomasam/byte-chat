"""
HOW A TOKENIZER WORKS -- explained with a tiny hand-checkable example.

This is Byte Pair Encoding (BPE), the method behind the tokenizers used by
GPT, Claude, Llama and essentially every modern language model.

THE PROBLEM
    A neural network can only do maths. It cannot read letters.
    So before anything else, text has to become numbers.

ATTEMPT 1 -- one number per letter
    Simplest idea: a=1, b=2, c=3 ... It works. But our 42-letter sentence
    then needs 42 numbers. That is wasteful: the model reads a fixed number
    of pieces at a time, so short pieces mean less text fits. We want FEWER
    pieces.

THE IDEA -- glue common pairs together
    Some pairs of letters turn up constantly: 'th', 'he', 'at'. Give each
    common pair its own number, and you need fewer numbers.

    How do we find which pairs are common?  We count.

    python3 tokeniser.py
"""
import collections

TEXT = "the cat sat on the mat the cat ate the rat"
ROUNDS = 6


def split_words(text):
    """STEP 1 -- chop the text into words.

    The spaces stay attached to the front of each word. That is how we can
    glue it back together later with no guesswork.

    Returns:
        list[str] -- one entry per word slot, repeats included.
        "the cat sat ..." -> ['the', ' cat', ' sat', ' on', ' the', ...]
    """
    return [(" " + word if index else word)
            for index, word in enumerate(text.split(" "))]


def count_words(words_list):
    """STEP 2 -- count how often each word appears.

    We store each word ONCE with a count, instead of a long list with
    repeats. Same information, less to walk through -- 11 slots become 8
    unique words here.

    Returns:
        Counter[str, int] -- word -> how many times it appears.
        Counter({' the': 3, ' cat': 2, 'the': 1, ' sat': 1, ...})
    """
    return collections.Counter(words_list)


def to_letters(counts):
    """STEP 3 -- break words into letters, keep the counts.

    Returns:
        dict[tuple[str, ...], int] -- letters of the word -> its count.
        {('t','h','e'): 1, (' ','c','a','t'): 2, ...}
    """
    return {tuple(word): count for word, count in counts.items()}


def train(words, rounds):
    """STEP 4 -- the merge loop  (this is the whole algorithm).

    Repeat:
        1. count every pair of NEIGHBOURS
        2. glue the most common pair into one piece

    A word's count rides along into every pair it contains, so the
    deduplication from STEP 2 costs nothing in accuracy.

    STEP 5 -- what comes out is the dictionary. That is all a "trained
    tokenizer" is: a list of pieces. No AI, no learning -- just counting
    and gluing.

    Returns:
        dict[str, int] -- piece -> its id, in the order they were learned.
        {'at': 256, 'th': 257, 'the': 258, ' the': 259, ' c': 260, ' cat': 261}
    """
    vocab = {}
    for round in range(1, rounds + 1):
        pair_counts = collections.Counter()
        for word, count in words.items():
            for pair in zip(word, word[1:]):
                pair_counts[pair] += count   # the word's count comes along so we don't have to revisit this 

        (left, right), _ = pair_counts.most_common(1)[0] # in real-life scenario most_common use a heap underneath to reduce the search complexity 
        glued = left + right
        vocab[glued] = 256 + round - 1

        merged_words = {}
        for word, count in words.items():
            index, pieces = 0, []
            while index < len(word):
                if (index < len(word) - 1
                        and word[index] == left
                        and word[index + 1] == right):
                    pieces.append(glued); index += 2 # because we fetched for most common pairs
                else:
                    pieces.append(word[index]); index += 1
            merged_words[tuple(pieces)] = count

        words = merged_words
    return vocab


def encode(text, vocab):
    """STEP 6 -- use it: glue pairs back together, earliest-learned first.

    Returns:
        list[str] -- the pieces text is made of; anything unlearned stays a
        single letter, so encoding never fails.
        "the cat" -> ['the', ' cat']
    """
    pieces = list(text)
    for piece in vocab:                       # vocab is in learned order
        index = 0
        while index < len(pieces) - 1:
            if pieces[index] + pieces[index + 1] == piece:
                pieces[index:index + 2] = [piece]
            else:
                index += 1
    return pieces


def main():
    """THE POINT.

    42 pieces (one per letter) become 21 pieces after 6 merges.

    A real tokenizer does exactly this, just bigger: ~32,503 rounds over
    ~2,000,000,000 characters, giving a dictionary of 32,768 pieces. The
    payoff: English text then needs about 4.8x fewer numbers than doing it
    letter by letter.

    That is the entire job:
        common words  ->  one number
        rare words    ->  a few numbers
        anything else ->  falls back to single letters, so nothing ever fails

    Returns:
        None -- prints the learned vocabulary, two worked encodings, and
        the before/after piece count.
    """
    words = to_letters(count_words(split_words(TEXT)))
    vocab = train(words, ROUNDS)

    for piece, piece_id in vocab.items():
        print(f"   {piece!r:>8}  =  {piece_id}")

    for sample in ["the cat", "the rat sat"]:
        pieces = encode(sample, vocab)
        ids = [vocab.get(piece, ord(piece[0])) for piece in pieces]
        print(f"\n   text    {sample!r}")
        print(f"   pieces  {pieces}")
        print(f"   numbers {ids}")
        print(f"   {len(sample)} letters -> {len(pieces)} pieces")

    print(f"\n   whole text: {len(TEXT)} letters -> "
          f"{len(encode(TEXT, vocab))} pieces after {ROUNDS} merges")


if __name__ == "__main__":
    main()
