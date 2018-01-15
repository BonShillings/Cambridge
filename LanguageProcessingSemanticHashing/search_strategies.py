# evaluate tf-idf etc to make document selections
import numpy as np
from scipy.spatial.distance import cosine
from operator import add

def tfidf_search(retrieval_selections,dataset,num_selections):
    tfidf_selections = []

    print ("Running TF-IDF")
    for index in range(len(retrieval_selections)):
        query = dataset[index]
        selected_indices = query_tfidf_search(query,index,retrieval_selections,dataset,num_selections)
        tfidf_selections.append(selected_indices)

    return tfidf_selections

def augmented_search(retrieval_selections, dataset, augmented_dataset, num_selections):
    tfidf_selections = []

    print ("Running TF-IDF")
    for index in range(len(retrieval_selections)):
        query = augmented_dataset[index]
        selected_indices = query_tfidf_search(query,index,retrieval_selections,dataset,num_selections)
        tfidf_selections.append(selected_indices)

    return tfidf_selections

# doubly managing the vector and document index so that prf_tfidf can fit into the search as well
def query_tfidf_search(vector, document_index,retrieval_selections,dataset,num_selections):
    preselection = retrieval_selections[document_index]
    preselected_vectors = dataset[preselection]

    vanilla_cosine_similarities = [1 - cosine(vector, preselected_vectors[i])
                                   for i in range(len(preselected_vectors))]

    # This probably sucks

    top_cosine_similarity_indices = np.argsort(vanilla_cosine_similarities)
    # reverse and select

    selected_indices = np.array(preselection)[top_cosine_similarity_indices][::-1]

    if len(selected_indices) > num_selections:
        selected_indices = selected_indices[:num_selections]

    return selected_indices

def prf_search(retrieval_selections,dataset,num_selections,prf_vector_selections, num_prf_docs=5, prf_ratio=1):
    prf_selections = []

    print ("Running PRF TF-IDF")
    for index in range(len(retrieval_selections)):
        base_vector = np.array(dataset[index])

        prf_indices = prf_vector_selections[index][:num_prf_docs]
        prf_vectors = np.array([dataset[prf_index] for prf_index in prf_indices])
        prf_update = prf_ratio * np.mean(prf_vectors, axis = 0)

        search_vector = base_vector + prf_update

        if index == 54 or index == 34 or index == 67:
            print len(base_vector)
            print len(search_vector)

        selected_indices = query_tfidf_search(search_vector, index, retrieval_selections, dataset, num_selections)
        prf_selections.append(selected_indices)

    return prf_selections

# maybe works
def prf_tfidf_search(retrieval_selections,dataset,num_selections,tfidf_selections=None, num_prf_docs=5, prf_ratio=1):
    if tfidf_selections is None:
        tfidf_selections= tfidf_search(retrieval_selections,dataset,num_prf_docs)

    prf_selections = prf_search(retrieval_selections,dataset,num_selections,tfidf_selections, num_prf_docs, prf_ratio)

    return prf_selections

# maybe works
def autoencoder_prf_tfidf_search(retrieval_selections,dataset,num_selections, encoder_selections=None, num_prf_docs=5, prf_ratio=1):

    #TODO
    if encoder_selections is None:
        encoder_selections= tfidf_search(retrieval_selections,dataset,num_prf_docs)

    prf_tfidf_selections = prf_search(retrieval_selections,dataset,num_selections,encoder_selections, num_prf_docs, prf_ratio)

    return prf_tfidf_selections
