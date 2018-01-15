from sklearn.datasets import fetch_rcv1
from sklearn.datasets import fetch_20newsgroups_vectorized,fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, TfidfTransformer

import numpy as np
np.set_printoptions(threshold=np.nan)

def build_file_rcv1():
    rcv1 = fetch_rcv1()

    fi = open("rcv1.txt","w")
    for sample in rcv1.data[1:5]:
        print(sample.toarray())
        fi.write(str(sample.toarray()) + '\n')
    fi.close()

def process_twenty_news(reduce_columns=10000,vectorized=False):
    twentynews_train = fetch_20newsgroups(subset='train')

    for i in range(50):
        article = twentynews_train.data[i]
        print i
        print article


    vectorizer = CountVectorizer(lowercase=True, tokenizer=None, stop_words="english", max_df=0.9, min_df=0.001, token_pattern="[a-zA-Z]+")
    twentynews_train.data = vectorizer.fit_transform(twentynews_train.data)
    twentynews_samples = twentynews_train.data.toarray()

    print twentynews_samples.shape
    totals = twentynews_samples[0]

    for sample in twentynews_samples:
        totals = np.add(totals,sample)

    ranks = np.argsort(totals)

    size = len(ranks)
    selections = []

    for i in range(len(ranks)):
        val = ranks[i]

        if val > (size - reduce_columns-1):
            selections.append(i)
            #print(totals[i])

    print(selections)
    print np.asarray(vectorizer.get_feature_names())[selections]

    #twentynews_train = fetch_20newsgroups(subset='train')
    twentynews_test = fetch_20newsgroups(subset='test') # this is fucked if we don't vectorize
    twentynews_test.data = vectorizer.transform(twentynews_test.data)

    #create reduced training / test data
    if vectorized:
        tfidftransformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
        #twentynews_train = fetch_20newsgroups_vectorized(subset='train')
        #twentynews_test = fetch_20newsgroups_vectorized(subset='test')
        # run count vectorizer
        twentynews_train.data = tfidftransformer.fit_transform(twentynews_train.data)
        twentynews_test.data = tfidftransformer.transform(twentynews_test.data)

    print twentynews_train.data.toarray().shape
    print twentynews_test.data.toarray().shape # BIG PROBLEM

    twentynews_train.data = twentynews_train.data[:,selections]
    twentynews_test.data = twentynews_test.data[:, selections]

    print twentynews_train.data.toarray().shape
    print twentynews_test.data.toarray().shape  # BIG PROBLEM

    print("Twenty news data processed and loaded")
    return twentynews_train,twentynews_test
    # return news_reduced_train.toarray(),news_reduced_test.toarray()

def process_twenty_news_vectors(reduce_columns=10000,vectorized=False):
    twentynews_train = fetch_20newsgroups(subset='train')
    tfidfvectorizer = TfidfVectorizer(lowercase=True, tokenizer=None, stop_words="english", max_df=0.9, min_df=0.001, token_pattern="[a-zA-Z]+")
    twentynews_train.data = tfidfvectorizer.fit_transform(twentynews_train.data)
    twentynews_samples = twentynews_train.data.toarray()

    print twentynews_samples.shape
    totals = twentynews_samples[0]

    for sample in twentynews_samples:
        totals = np.add(totals,sample)

    ranks = np.argsort(totals)

    size = len(ranks)
    selections = []

    for i in range(len(ranks)):
        val = ranks[i]

        if val > (size - reduce_columns-1):
            selections.append(i)

    print np.asarray(tfidfvectorizer.get_feature_names())[selections]

    twentynews_test = fetch_20newsgroups(subset='test') # this is fucked if we don't vectorize

    twentynews_test.data = tfidfvectorizer.transform(twentynews_test.data)
    print twentynews_train.data.toarray().shape
    print twentynews_test.data.toarray().shape # BIG PROBLEM

    twentynews_train.data = twentynews_train.data[:,selections]
    twentynews_test.data = twentynews_test.data[:, selections]

    print twentynews_train.data.toarray().shape
    print twentynews_test.data.toarray().shape  # BIG PROBLEM

    print("Twenty news data processed and loaded")
    return twentynews_train,twentynews_test
    # return news_reduced_train.toarray(),news_reduced_test.toarray()



