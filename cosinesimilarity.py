import numpy as np
from math import sqrt, log
from itertools import chain, product
from collections import defaultdict

def cosine_sim(u,v):
    return np.dot(u,v) / (sqrt(np.dot(u,u)) * sqrt(np.dot(v,v)))

def ngrams(sentence, n):
  return zip(*[sentence.split()[i:] for i in range(n)])

def tfidf(corpus, vocab):
    """
    INPUT:

    corpus = [('this is a foo bar', [1, 1, 0, 1, 1, 0, 0, 1]), 
    ('foo bar bar black sheep', [0, 2, 1, 1, 0, 0, 1, 0]), 
    ('this is a sentence', [1, 0, 0, 0, 1, 1, 0, 1])]

    vocab = ['a', 'bar', 'black', 'foo', 'is', 'sentence', 
    'sheep', 'this']

    OUTPUT:

    [[0.300, 0.300, 0.0, 0.300, 0.300, 0.0, 0.0, 0.300], 
    [0.0, 0.600, 0.600, 0.300, 0.0, 0.0, 0.600, 0.0], 
    [0.375, 0.0, 0.0, 0.0, 0.375, 0.75, 0.0, 0.375]]

    """
    def termfreq(matrix, doc, term):
        try: return matrix[doc][term] / float(sum(matrix[doc].values()))
        except ZeroDivisionError: return 0
    def inversedocfreq(matrix, term):
        try: 
            return float(len(matrix)) /sum([1 for i,_ in enumerate(matrix) if matrix[i][term] > 0])
        except ZeroDivisionError: return 0

    matrix = [{k:v for k,v in zip(vocab, i[1])} for i in corpus]
    tfidf = defaultdict(dict)
    for doc,_ in enumerate(matrix):
        for term in matrix[doc]:
            tf = termfreq(matrix,doc,term)
            idf = inversedocfreq(matrix, term)
            tfidf[doc][term] = tf*idf

    return [[tfidf[doc][term] for term in vocab] for doc,_ in enumerate(tfidf)]


def corpus2vectors(corpus):
    def vectorize(sentence, vocab):
        return [sentence.split().count(i) for i in vocab]
    vectorized_corpus = []
    vocab = sorted(set(chain(*[i.lower().split() for i in corpus])))
    for i in corpus:
        vectorized_corpus.append((i, vectorize(i, vocab)))
    return vectorized_corpus, vocab

def create_test_corpus():
    sent1 = "this is a foo bar"
    sent2 = "foo bar bar black sheep"
    sent3 = "this is a sentence"

    all_sents = [sent1,sent2,sent3]
    corpus, vocab = corpus2vectors(all_sents)
    return corpus, vocab

def test_cosine():
    corpus, vocab = create_test_corpus()

    for sentx, senty in product(corpus, corpus):
        print sentx[0]
        print senty[0]
        print "cosine =", cosine_sim(sentx[1], senty[1])
        print

def test_ngrams():
    corpus, vocab = create_test_corpus()
    for sentx in corpus:
        print sentx[0]
        print ngrams(sentx[0],2)
        print ngrams(sentx[0],3)
        print

def test_tfidf():
    corpus, vocab = create_test_corpus()
    print corpus
    print vocab
    print tfidf(corpus, vocab)

print "Testing cosine..."
test_cosine()
print
print "Testing ngrams..."
test_ngrams()
print
print "Testing tfidf..."
test_tfidf()
print