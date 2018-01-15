import data_processing
import bit_utilities
import evaluation_utilities

import TrainAutoencoderKeras
import operator

import numpy as np
np.set_printoptions(threshold=np.nan)

import search_strategies

from scipy.spatial.distance import cosine

from sklearn.metrics import pairwise

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist

column_size = 10000 #hyper
vectorized = True # hyper
noise = 2.0 # hyper

full_news_train, full_news_test = data_processing.process_twenty_news_vectors(column_size,vectorized=vectorized)

news_train = full_news_train.data.toarray()
news_test = full_news_test.data.toarray()

# train/test
dataset = news_test
full_dataset = full_news_test

#add noise
noise_factor = 0.0
news_train_noisy = news_train + noise_factor * np.random.normal(loc=0.0, scale=noise, size=news_train.shape)
news_test_noisy = news_test + noise_factor * np.random.normal(loc=0.0, scale=noise, size=news_test.shape)

if vectorized:
    news_train_noisy = np.clip(news_train_noisy, 0., 1.)
    news_test_noisy = np.clip(news_test_noisy, 0., 1.)

print news_train_noisy.shape
print news_test_noisy.shape

network_structure = [500,500,32,500,500,column_size] # hyper

autoencoder = TrainAutoencoderKeras.build_autoencoder_stack(layers=network_structure,num_encoders=3,x_train=news_train_noisy)

'''
validation_data=None
if news_test is not None:
    validation_data = (news_test_noisy, news_test)

autoencoder.fit(news_train_noisy, news_train, epochs=20, batch_size=256, shuffle=True
                #,validation_data=(news_test_noisy,news_test)
                )

print(autoencoder.summary())

autoencoder.save_weights("20_SemanticHashingWeights.txt",overwrite=True)
'''
autoencoder.load_weights("SemanticHashingWeights.txt")

encoder = TrainAutoencoderKeras.extract_encoder_from_file(filepath="SemanticHashingWeights.txt",
                                                          layers=network_structure, num_encoders=3, x_train=news_train, )

print(encoder.summary())

encodings = encoder.predict(dataset)

bit_encodings = [bit_utilities.convert_encoding_to_bits(np.where(encoding > 0.5, 1, 0)) for encoding in encodings]
# encode vectors

encoding_to_vectors = {}

for i in range(1,50):
    print np.mean(news_train[i])
    bit_code = bit_encodings[i]
    print(encodings[i])
    print(np.where(encodings[i] > 0.5, 1, 0))
    print(bin(bit_code))
    # sanity check on encoder activations


# generate encoding to indices map

encoding_to_indices = {}

for i in range(len(bit_encodings)):
    encoding = bit_encodings[i]
    if encoding in encoding_to_vectors:
        encoding_to_vectors[encoding].append(news_train[i])
        encoding_to_indices[encoding].append(i)
    else:
        encoding_to_vectors[encoding] = [news_train[i]]
        encoding_to_indices[encoding] = [i]

print(len(encoding_to_vectors[bit_encodings[50]]))

# generate pre-selections from autoencoder

hamming_distance = 6
retrieval_selections = []

encoding_to_hammig_set = {}
for e in range(len(bit_encodings)):
    if e % 100 == 0:
        print (e)

    encoding = bit_encodings[e]
    # compute vectors in hamming distance 6

    preselected_vector_indices = bit_utilities.retrieve_docs_within_hamming_distance(encoding,encoding_to_indices,hamming_distance)

    if e not in preselected_vector_indices:
        print("Encoding error for index: " + str(e))
        print (encoding)

    retrieval_selections.append(preselected_vector_indices)  # unsorted

    # could evaluate on the fly to save memory

targets = full_dataset.target
num_selections = 100 #hyper


print("Encoded Density")
evaluation_utilities.evaluate_precisions(retrieval_selections,targets)


print("Vanilla TF-IDF")
tfidf_selections = search_strategies.tfidf_search(retrieval_selections,dataset,num_selections)

evaluation_utilities.evaluate_precisions(tfidf_selections,targets)

precision_matrix, precision_means = evaluation_utilities.evaluate_precision_recall(tfidf_selections,targets,num_selections)

for i in range(len(precision_means)):
    print("TFIDF Precision at " + str(i+1) + " " + str(precision_means[i]))

'''
# evaluate pseudo relevance feedback
prf_tfidf_selections = search_strategies.prf_tfidf_search(retrieval_selections,dataset,num_selections,tfidf_selections,prf_ratio=1,num_prf_docs=10)

precision_matrix, precision_means = evaluation_utilities.evaluate_precision_recall(prf_tfidf_selections,targets,num_selections)

for i in range(len(precision_means)):
    print("PRF precision at " + str(i+1) + " " + str(precision_means[i]))

# evaluate autoencoder pseudo relevance feedback
# TODO: build a-prf
'''

# evaluate reconstruction search


reconstructed_dataset = autoencoder.predict(dataset)

'''
print("Reconstruction search")
reconstruction_search_retrievals = search_strategies.augmented_search(retrieval_selections, dataset, reconstructed_dataset, num_selections)

precision_matrix, precision_means = evaluation_utilities.evaluate_precision_recall(reconstruction_search_retrievals,targets,num_selections)

for i in range(len(precision_means)):
    print("Reconstruction precision at " + str(i+1) + " " + str(precision_means[i]))
'''

# evaluate gradient search augmentation

alphas = [1,5,10,15]

print("Negative GSA")

for alpha in alphas:
    print("\n")
    print(alpha)

    log_p_gradient = alpha*(reconstructed_dataset - dataset) / noise**2
    print(log_p_gradient.shape)

    gradient_augmented_dataset = dataset - log_p_gradient
    print(gradient_augmented_dataset.shape)

    print("Negative augmented GSA")

    #actually kind of wrong
    gradient_augmented_retrievals = search_strategies.augmented_search(retrieval_selections, dataset, gradient_augmented_dataset, num_selections)

    precision_matrix, precision_means = evaluation_utilities.evaluate_precision_recall(gradient_augmented_retrievals,targets,num_selections)
    for i in range(len(precision_means)):
        print("Negative augmented GSA precision at " + str(i + 1) + " " + str(precision_means[i]))

    print("Negative tfidf GSA")

    # actually kind of wrong
    gradient_augmented_retrievals = search_strategies.tfidf_search(retrieval_selections,
                                                                       gradient_augmented_dataset, num_selections)

    precision_matrix, precision_means = evaluation_utilities.evaluate_precision_recall(gradient_augmented_retrievals,
                                                                                     targets, num_selections)
    for i in range(len(precision_means)):
        print("Negative tfidf GSA precision at " + str(i + 1) + " " + str(precision_means[i]))

    print("Positive")
    gradient_augmented_dataset = dataset + log_p_gradient

    print(gradient_augmented_dataset.shape)

    print("Positive augmented GSA")

    # actually kind of wrong
    gradient_augmented_retrievals = search_strategies.augmented_search(retrieval_selections, dataset,
                                                                       gradient_augmented_dataset, num_selections)

    precision_matrix, precision_means = evaluation_utilities.evaluate_precision_recall(gradient_augmented_retrievals,
                                                                                       targets, num_selections)
    for i in range(len(precision_means)):
        print("Positive augmented GSA precision at " + str(i + 1) + " " + str(precision_means[i]))

    print("Positive tfidf GSA")

    # actually kind of wrong
    gradient_augmented_retrievals = search_strategies.tfidf_search(retrieval_selections,
                                                                   gradient_augmented_dataset, num_selections)

    precision_matrix, precision_means = evaluation_utilities.evaluate_precision_recall(gradient_augmented_retrievals,
                                                                                       targets, num_selections)

    for i in range(len(precision_means)):
        print("Positive tfidf GSA precision at " + str(i + 1) + " " + str(precision_means[i]))

'''
print("GSA + PRF")

#actually kind of wrong
gradient_augmented_prf_retrievals = search_strategies.prf_tfidf_search(retrieval_selections,gradient_augmented_dataset,num_selections,tfidf_selections,prf_ratio=1,num_prf_docs=10)

precision_matrix, precision_means = evaluation_utilities.evaluate_precision_recall(gradient_augmented_prf_retrievals,targets,num_selections)

for i in range(len(precision_means)):
    print("PRF+GSA precision at " + str(i+1) + " " + str(precision_means[i]))
'''


