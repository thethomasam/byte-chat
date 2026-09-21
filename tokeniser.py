"""
HOW A TOKENIZER WORKS -- explained with a tiny hand-checkable example.

This is Byte Pair Encoding (BPE), the method behind the tokenizers used by
GPT, Claude, Llama and essentially every modern language model.

    python3 explain_tokenizer.py
"""
import collections

TEXT = "the cat sat on the mat the cat ate the rat"
ROUNDS = 6

def title(t):
    print(f"\n\n{'='*64}\n  {t}\n{'='*64}")


title("THE PROBLEM")
print("""
A neural network can only do maths. It cannot read letters.
So before anything else, text has to become numbers.
""")
print(f'   our text:  "{TEXT}"')


title("ATTEMPT 1 -- one number per letter")
print("""
Simplest idea: a=1, b=2, c=3 ... It works. But look how many
numbers you need for such a short sentence.
""")
print(f"   {len(TEXT)} letters  ->  {len(TEXT)} numbers")
print(f"   {list(TEXT[:14])} ...")
print("""
That is wasteful. The model reads a fixed number of pieces at a
time, so short pieces mean less text fits. We want FEWER pieces.
""")


title("THE IDEA -- glue common pairs together")
print("""
Some pairs of letters turn up constantly: 'th', 'he', 'at'.
Give each common pair its own number, and you need fewer numbers.

How do we find which pairs are common?  We count.
""")


title("STEP 1 -- chop the text into words")
words_list = [(" " + w if i else w) for i, w in enumerate(TEXT.split(" "))]
print(f"\n   {words_list}")
print("""
   Note the spaces stay attached to the front of each word.
   That is how we can glue it back together later with no guesswork.
""")


title("STEP 2 -- count how often each word appears")
counts = collections.Counter(words_list)
for w, n in counts.most_common():
    print(f"   {w!r:>8}  x{n}")
print("""
   We store each word ONCE with a count, instead of a long list
   with repeats. Same information, less to walk through.
""")


title("STEP 3 -- break words into letters, keep the counts")
words = {tuple(w): n for w, n in counts.items()}
for w, n in list(words.items())[:4]:
    print(f"   {list(w)}  x{n}")
print("   ...")


title("STEP 4 -- the merge loop  (this is the whole algorithm)")
print("""
   Repeat:
       1. count every pair of NEIGHBOURS
       2. glue the most common pair into one piece
""")
vocab = {}
for r in range(1, ROUNDS + 1):
    pairs = collections.Counter()
    for w, n in words.items():
        for pair in zip(w, w[1:]):
            pairs[pair] += n              # <- +n : the word's count comes along

    (a, b), n = pairs.most_common(1)[0]
    glued = a + b
    vocab[glued] = 256 + r - 1

    top = ", ".join(f"{x+y!r}:{c}" for (x, y), c in pairs.most_common(3))
    print(f"\n   ROUND {r}   most common pairs -> {top}")
    print(f"           WINNER {a!r}+{b!r} seen {n}x  ->  new piece {glued!r}")

    new = {}
    for w, c in words.items():
        i, out = 0, []
        while i < len(w):
            if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                out.append(glued); i += 2
            else:
                out.append(w[i]); i += 1
        new[tuple(out)] = c
    words = new
    print(f"           text is now: {[list(w) for w in list(words)[:3]]} ...")


title("STEP 5 -- the dictionary we just built")
print("""
   That is all a "trained tokenizer" is. A list of pieces.
   No AI, no learning -- just counting and gluing.
""")
for piece, num in vocab.items():
    print(f"   {piece!r:>8}  =  {num}")


title("STEP 6 -- use it")
def encode(s):
    """glue pairs back together, earliest-learned first"""
    out = list(s)
    for piece in vocab:                       # vocab is in learned order
        i = 0
        while i < len(out) - 1:
            if out[i] + out[i + 1] == piece:
                out[i:i + 2] = [piece]
            else:
                i += 1
    return out

for s in ["the cat", "the rat sat"]:
    pieces = encode(s)
    ids = [vocab.get(p, ord(p[0])) for p in pieces]
    print(f"\n   text    {s!r}")
    print(f"   pieces  {pieces}")
    print(f"   numbers {ids}")
    print(f"   {len(s)} letters -> {len(pieces)} pieces")


title("THE POINT")
before, after = len(TEXT), len(encode(TEXT))
print(f"""
   Before:  {before} pieces   (one per letter)
   After:   {after} pieces   (after {ROUNDS} merges)

   A real tokenizer does exactly this, just bigger:
       {32503:,} rounds instead of {ROUNDS}
       2,000,000,000 characters instead of {len(TEXT)}
       a dictionary of 32,768 pieces instead of {ROUNDS}

   The payoff: English text then needs about 4.8x fewer numbers
   than doing it letter by letter.

   That is the entire job:
       common words  ->  one number
       rare words    ->  a few numbers
       anything else ->  falls back to single letters, so nothing ever fails
""")
