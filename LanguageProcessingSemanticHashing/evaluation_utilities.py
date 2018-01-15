import numpy as np

def evaluate_precisions(retrieval_selections, targets):
    precisions = []

    sum_len = 0.0
    for i in range(len(retrieval_selections)):
        retrieval = retrieval_selections[i]
        target = targets[i]
        correct_count = 0.0
        for retrieved_index in retrieval:
            if targets[retrieved_index] == target:
                correct_count += 1

        precision = correct_count / float(len(retrieval))

        precisions.append(precision)

        sum_len += len(retrieval)

    #print(precisions)
    print np.mean(precisions)

    avg_len = sum_len / len(retrieval_selections)
    print(avg_len)

    return np.mean(precisions)

#       precision 1    precision 2
# doc 1
# doc 2
def evaluate_precision_recall(retrieval_selections,targets,result_limit):
    precision_matrix = [[] for x in retrieval_selections]
    recall_matrix = [[] for x in retrieval_selections]
    for i in range(len(retrieval_selections)):
        retrieval = retrieval_selections[i]
        limit = min(result_limit,len(retrieval))
        target = targets[i]
        correct_count = 0.0

        for j in range(limit):
            retrieved_index = retrieval[j]
            if targets[retrieved_index] == target:
                correct_count += 1

            precision = correct_count / float(j+1)
            precision_matrix[i].append(precision)
        #print(precision_matrix[i])

    precision_means = []
    for i in range(result_limit):
        precisions_at_i = []
        for j in range(len(precision_matrix)):
            document_precisions = precision_matrix[j]
            #print(document_precisions)
            if len(document_precisions) > i:
                precisions_at_i.append(document_precisions[i])
        precision_means.append(np.mean(precisions_at_i))

    return precision_matrix, precision_means