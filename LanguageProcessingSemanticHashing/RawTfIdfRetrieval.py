import data_processing
import bit_utilities
import search_strategies
import evaluation_utilities

import numpy as np
np.set_printoptions(threshold=np.nan)

from sklearn.metrics import pairwise


full_news_train, full_news_test = data_processing.process_twenty_news(10000,vectorized=True)

news_train = full_news_train.data.toarray()
news_test = full_news_test.data.toarray()

dataset = news_test
num_selections = 100
retrieval_selections = np.array([range(len(dataset)) for _ in dataset])

print dataset.shape
print retrieval_selections.shape

tfidf_selections = search_strategies.tfidf_search(retrieval_selections,dataset,num_selections)

targets = full_news_train.target

evaluation_utilities.evaluate_precision_recall(tfidf_selections,targets,num_selections)

