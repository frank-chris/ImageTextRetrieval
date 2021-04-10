"""
Constructing and loading dictionaries
"""
import numpy
from collections import OrderedDict

def build_dictionary(text):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """

    # print(text)
    wordcount = OrderedDict()
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1
    words = list(wordcount.keys())
    freqs = list(wordcount.values())

    print(words[:10])

    sorted_idx = numpy.argsort(freqs)[::-1]

    worddict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        worddict[words[sidx]] = idx + 2   # 0: <eos>, 1: <unk>

    return worddict, wordcount