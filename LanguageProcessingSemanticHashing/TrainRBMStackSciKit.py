import numpy as np
from sklearn.neural_network import BernoulliRBM




X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])



model = BernoulliRBM(n_components=2)


model.fit(X)
BernoulliRBM(batch_size=10, learning_rate=0.1, n_components=2, n_iter=10,
       random_state=None, verbose=0)

'''
data: input layer of network
network_structure: array of sizes of each layer of network
'''
def train_rbm_stack(data,network_structure, batch_size=10, learning_rate=0.1,n_iter=10,random_state=None,verbose=0):
       weights = []
       visible_unit_samples = data
       for layer in network_structure:

              model = BernoulliRBM(n_components=layer, batch_size=batch_size, learning_rate=learning_rate,
                                          n_iter=n_iter, random_state=random_state, verbose=verbose)

              hidden_unit_samples = model.fit_transform(visible_unit_samples)

              weights.append(model.components_)

              visible_unit_samples = hidden_unit_samples

       return weights


