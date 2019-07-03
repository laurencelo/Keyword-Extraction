import string
import nltk
import collections
import math
from nltk.chunk.regexp import *
from rake_nltk import Rake
from nltk.collocations import *
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import *

# Read input as string and token
input_text = open("InputText1.txt", 'r').read()
input_tokens = wordpunct_tokenize(input_text)


def removeStopWords(text_tokens):
    stopwords_set = set(stopwords.words("english")).union(set(string.punctuation))
    stopwords_set = stopwords_set.union(set('’')).union(set('“')).union(set('”'))
    result = [word for word in text_tokens if not word in stopwords_set]
    return result


def stemming(text_tokens):
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    stemmed_words = []
    for token in text_tokens:
        stemmed_words.append(stemmer.stem(token))
    return stemmed_words


def removePunctuation(sentence):
    punc_set = set(string.punctuation).union(set('“')).union(set('”')).union(set('’'))
    result = ''.join(ch for ch in sentence if ch not in punc_set)
    return result


# Method 1: extract the most frequent n nouns by POS tagging
def Method1(n=5):
    M1_input = removeStopWords(input_tokens)
    pos_tags = nltk.pos_tag(M1_input, tagset='universal')
    noun_words = []
    for pair in pos_tags:
        if pair[1] == 'NOUN':
            noun_words.append(pair[0])
    keywords = collections.Counter(noun_words).most_common(n)
    for pair in keywords:
        print(pair[0])


# Method 2: extract multiple keywords from the most frequent n collocations
def Method2(n=5):
    M2_input = removeStopWords(input_tokens)
    bgam = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(M2_input)
    collocations = finder.nbest(bgam.likelihood_ratio, n)
    for c in collocations:
        print(' '.join(c))


# Method 3: extract most frequent n noun phrases
def Method3(n=5):
    M3_input = sent_tokenize(input_text)
    np_grammar = "NP: {<DET>?<ADJ>*<NOUN>+}"
    chunk_parser = RegexpParser(np_grammar)
    NP = []
    for sent in M3_input:
        tokens = word_tokenize(sent)
        tag_tokens = nltk.pos_tag(tokens, tagset='universal')
        result = chunk_parser.parse(tag_tokens)
        for subtree in result.subtrees():
            if subtree.label() == "NP":
                words = " ".join([x for (x, y) in subtree.leaves()])
                NP.append(words)
    keywords = collections.Counter(NP).most_common(n)
    for data in keywords:
        print(data[0])


# Method 4: rake (rapid automatic keyword extraction)
def Method4(n=5):
    M4_input = input_text
    r = Rake()
    r.extract_keywords_from_text(M4_input)
    r.get_ranked_phrases()
    for key in r.get_ranked_phrases_with_scores()[:n]:
        print(removePunctuation(key[1].strip()))


# Method 5: TD-IDF method
def Method5(n=5):
    M5_input = sent_tokenize(input_text)

    # TF: fij = frequency of term i in sentence j, Nj = total number of words in sentence j
    fij = []
    Nj = []
    TF_result = []
    for sen in M5_input:
        tokens = word_tokenize(sen)
        word_freq = dict(collections.Counter(tokens))
        fij.append(word_freq)
        Nj.append(len(tokens))
    # Compute TF value
    for i in range(len(Nj)):
        tokens = word_tokenize(M5_input[i])
        temp = {}
        for word in tokens:
            temp[word] = fij[i].get(word) / Nj[i]
        TF_result.append(temp)

    # IDF: N = number of sentences, ni = number of sentences mention word i
    N = len(M5_input)
    IDF_result = {}
    for sen in M5_input:
        tokens = word_tokenize(sen)
        for word in tokens:
            ni = 0
            for s in M5_input:
                if word in s:
                    ni += 1
                    continue
            IDF_result[word] = math.log(N / ni)

    # Compute TF-IDF value
    TF_IDF = {}
    for sent_TF in TF_result:
        for word, value in sent_TF.items():
            TF_IDF[word] = value * IDF_result[word]
    TF_IDF_sort = sorted(TF_IDF.items(), key=lambda x: x[1], reverse=True)
    for word, value in TF_IDF_sort[:n]:
        print(word)

# Method1()
# Method2()
# Method3()
# Method4()
# Method5()
