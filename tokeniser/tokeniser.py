"""
This is my simple explanation on how tokenisers work. We know there is a lot of jargon around how LLM work behind the curtain. 
Bamm!! ths is the first step towards the jagron."Tokenisation".

The idea behind tokenisation is Large Language Models(LLMs) don't understand words so we need to figure out a way to convert 
these words to number so that the model understands the semantic meaning behind words and memorise their relationship;p

So tokenisations does this by assigning token a number to each char so that means if I want to say "i love llms" 
and each char has a number assigned then 
1=23
l=45..... and so far so on

[23, 34, 45,46, 56,58,........] so that would cause a long list of numbers to accommodate a simple sentence. 

Therefore a smarter way is to learn the pattern of what words come together and assign them a token number so like if I have corpus of things i love

1. i love llms
2. i love mamba
3. i love grasshpper
4. sometimes love lion
5. I hate icecream

we might observe a pattern that "i love" comes together a lot so why not assign them a a token number
thats essentially what Byte Pair Encoding does.

"""
import collections

TEXT = "the cat sat on the mat the cat ate the rat"
ROUNDS = 6


def split_words(text):
    """STEP 1 -- chop the text into words.

    Returns:
        list[str] -- one entry per word slot, repeats included.
        "the cat sat ..." -> ['the', ' cat', ' sat', ' on', ' the', ...]
    """
    return [(" " + word if index else word)
            for index, word in enumerate(text.split(" "))]


def count_words(words_list):
    """STEP 2 -- count how often each word appears.

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
    """STEP 4 -- merge the chars they apear together 

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
        print(f"   {piece}  =  {piece_id}")

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
