from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import os

import pickle
import json

#from pymix import mixture

MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19
}

# initialize the input image shape (224x224 pixels) along with
# the pre-processing function (this might need to be changed
# based on which model we use to classify our image)
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

print("[INFO] loading {}...".format("vgg16"))
Network = MODELS["vgg16"]
model = Network(weights="imagenet")


convolution_model = Model(input=model.input, output=model.layers[-4].output)
convolution_model.summary()
model.summary()
# Classify Image
# load the input image using the Keras helper utility while ensuring
# the image is resized to `inputShape`, the required input dimensions
# for the ImageNet pre-trained network

all_labels = {}

filenames_to_activations = {}
all_filenames = []
all_activations = []
directory = '/Users/sean/Cambridge/yelp_photos/photos_large'

total_files = len(os.listdir(directory))
print total_files
count = 0.0
advance = 1.0
threshold = 100
filename_to_activation_labels = {}

for filename in os.listdir(directory):
    if not 'DS_Store' in filename:

        print("[INFO] loading and pre-processing image..." + filename)
        image = load_img(directory + "/" + filename, target_size=inputShape)
        image = img_to_array(image)

        # our input image is now represented as a NumPy array of shape
        # (inputShape[0], inputShape[1], 3) however we need to expand the
        # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
        # so we can pass it through thenetwork
        image = np.expand_dims(image, axis=0)

        # pre-process the image using the appropriate function based on the
        # model that has been loaded (i.e., mean subtraction, scaling, etc.)
        image = preprocess(image)

        print("[INFO] classifying image with '{}'...".format('vgg16'))
        activations = convolution_model.predict(image)[0]
        #P = imagenet_utils.decode_predictions(preds)

        filename_to_activation_labels[filename] = []
        for i in range(len(activations)):
            if activations[i] > threshold:
                filename_to_activation_labels[filename].append(str(i))

        #could sample as binary variable

        filenames_to_activations[filename] = activations
        all_filenames.append(filename)
        all_activations += activations.tolist()

        count += 1.0
        # print count/total_files
        if advance < 100 * (count / total_files):
            print "{} percent complete".format(count / total_files)
            advance += 1

total_images = len(filename_to_activation_labels)
print total_images

with open('image_activations.json', 'w') as fp:
    json.dump(filename_to_activation_labels, fp)

#print filename_to_activation_labels

# Importing Gensim
import gensim
from gensim import corpora

print np.array(filename_to_activation_labels.values()).shape

# Creating the term dictionary of our courpus, where every unique term is assigned an index.



dictionary = corpora.Dictionary(filename_to_activation_labels.values())

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in filename_to_activation_labels.values()]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

num_topics = 5
# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=num_topics, id2word = dictionary, passes=100)

for topic in ldamodel.print_topics(num_topics=num_topics, num_words=10):
	print(topic)

with open('/Users/sean/Cambridge/yelp_dataset/photos.txt', 'r') as f:
    image_json_dict = json.load(f)

topic_to_label_counts = {0:{},1:{},2:{},3:{},4:{}}
topic_to_image = {0:[],1:[],2:[],3:[],4:[]}

for image_json in image_json_dict:

    img_name = image_json["photo_id"] + ".jpg"
    if img_name in filename_to_activation_labels:
        doc = filename_to_activation_labels[img_name]
        label = "none"
        if "label" in image_json:
            label = image_json["label"]

        # test
        topic_scores = ldamodel[dictionary.doc2bow(doc)]
        #print topic_scores
        #print label
        top_prob = 0.0
        top_topic = 0
        #pick most likely topic
        for topic, prob in topic_scores:
            if prob > top_prob:
                top_topic = topic
                top_prob = prob
        if label in topic_to_label_counts[top_topic]:
            topic_to_label_counts[top_topic][label] += 1
        else:
            topic_to_label_counts[top_topic][label] = 1

print topic_to_label_counts

for topic_sample in topic_to_image:
    for i in range(10):
        print topic_to_image[topic_sample][i]
'''
data = mixture.DataSet()
data.fromList(all_activations,IDs=all_filenames)

print all_activations[0]
num_activations = len(all_activations[0])
print num_activations
prior = [1.0/num_activations] * num_activations

d1 = mixture.DiscreteDistribution(num_activations,prior)
d2 = mixture.DiscreteDistribution(num_activations,prior)
d3 = mixture.DiscreteDistribution(num_activations,prior)
d4 = mixture.DiscreteDistribution(num_activations,prior)

mix = mixture.MixtureModel(4,[0.25,0.25,0.25,0.25],[d1,d2,d3,d4])

example = mix.sampleDataSet(1)
print example

mix.EM(data,40,0.1)
'''