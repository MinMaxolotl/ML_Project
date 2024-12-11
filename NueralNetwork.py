import sklearn.metrics
import tensorflow as tf
import keras
from keras import datasets, layers, models
import sklearn
import numpy as np
import matplotlib.pyplot as plt


# Creates a Feed-Forward Neural Network
# Inputs: training_pairs, shape (num_pairs, features_per_song*2)
# Outputs: model object
def makeFFNN(num_features):
    model = models.Sequential([
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # use the goated adam optimizer, along with cosine similarity loss metric
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses 
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/CosineSimilarity 
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # input shape must have the same number of features as the training data has
    # by defining the shape as a tuple with only one input, keras just looks that the input has
    # "training_data.shape[1]" many features

    shape = (None, num_features)
    model.build(input_shape=shape)
    print(model.summary())

    return model


# helper function for trainFFNN to create similarity pairs in data so that the model is provided
# with labeled examples of what two similar songs look like so that the model learns to find similarities in songs
# Inputs: training_data, shape: (n_songs, features_per_song)
# Outputs: training_pairs, shape (num_pairs, features_per_song*2)
#          training_labels, shape (num_pairs, 1), 1 = similar, 0 = not similar
def makeTrainingPairs(training_data, playlist_data):
    num_songs = training_data.shape[0]
    num_playlist_songs = playlist_data.shape[0]
    training_pairs = []
    training_labels = []

    # use euclidian distance to solve for distance between every feature set from every other feautre set
    # https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html
    eucl_dist = sklearn.metrics.euclidean_distances(training_data)

    # normalize euclidian distance
    max_val = np.max(eucl_dist)
    min_val = np.min(eucl_dist)
    eucl_dist = (eucl_dist - min_val) / (max_val - min_val)

    margin = np.mean(eucl_dist)

    # Compare every song with every other song. If distance between two are larger than margin, they are labeled not similar
    for i in range(num_songs):
        for j in range(num_playlist_songs):
            # if i == j: # for situation where we are comparing a song to itself
            #     continue
            d = eucl_dist[i,j]
            
            if d > margin:
                training_labels.append(0)
            else:
                training_labels.append(1)

            # concatenate the feature sets together
            training_pairs.append(np.concatenate([training_data[i], playlist_data[j]]))

    print(np.unique(training_labels, return_counts=True))

    return np.array(training_pairs), np.array(training_labels)

def trainFFNN(model, training_data, playlist_data, epochs):
    training_pairs, training_labels = makeTrainingPairs(training_data, playlist_data)

    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit 
    history_activations = model.fit(training_pairs, training_labels, epochs=epochs, batch_size=32)

    # plot accuracy change over epochs
    for i in range(epochs):
        plt.plot(history_activations.history['accuracy'], 'o-')
    plt.title('Feedforward Neural Network Training Accuracy')
    plt.ylabel('training accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.show()

    # now save model so we dont need to retrain
    model.save("./trained_model.h5")

    return model

# helper function that creates pairs of playlist songs and test dataset songs
# Inputs:  testing_data, shape: (n_songs, features_per_song)
#          playlist_data, shape: (playlist_size, features_per_song)
# Outputs: inference_pairs, shape (num_pairs, features_per_song*2)
def makeInferencePairs(testing_data, playlist_data):
    inference_pairs = []
    num_test_songs = testing_data.shape[0]
    playlist_size = playlist_data.shape[0]

    for i in range(num_test_songs):
        for j in range(playlist_size):
            combined_features = np.concatenate([testing_data[i], playlist_data[j]])
            inference_pairs.append(combined_features)

    return np.array(inference_pairs)

def inference(model, testing_data, playlist_data, output_playlist_size):
    # get inference pairs
    inference_pairs = makeInferencePairs(testing_data, playlist_data)

    # now we let the model solve for what the similarities are between the playlists
    similarities = model.predict(inference_pairs)

    # to map the similarities to a shape where we can check the pairs of testing data and the playlist, we can
    # remap the similarities vector which is a shape of (n_testing_songs * n_playlist_songs, 1) to (n_testing_songs, n_playlist_songs)
    # then if i want to know how similar song 3 in the test is to song 10 in the playlist, i can call matrix[3][10] and get the similarity val
    similarities = np.reshape(similarities, (testing_data.shape[0], playlist_data.shape[0]))

    # now we sum the similarity scores for every test dataset song for every song in the playlist
    # in other words, if every row is the similarity scores between one test dataset song and every playlist song,
    # then if we sum that row together into one value, we get the total similarity score
    sum_similarities = np.sum(similarities, axis=1)
    
    # organize songs based on value of sums, reverse the list so indices correlated to largest number comes first
    # https://stackoverflow.com/questions/6771428/most-efficient-way-to-reverse-a-numpy-array 
    top_songs_index = np.argsort(sum_similarities)[::-1]

    # now grab the first N indexes
    top_songs_index = top_songs_index[:output_playlist_size]
    
    return top_songs_index


def modelEvaluate(model, testing_data, playlist_data):
    # make pairs for testing
    testing_pairs, testing_labels = makeTrainingPairs(testing_data, playlist_data)

    # create untrained model and find accuracy on testing pairs
    untrained_model = makeFFNN(playlist_data.shape[1]*2)
    _, untrained_acc = untrained_model.evaluate(testing_pairs, testing_labels)
    
    # check accuracy of trained model on testing pairs
    _, accuracy = model.evaluate(testing_pairs, testing_labels)

    # create square bar plot of untrained vs trained accuracy
    plt.figure(figsize=(8, 8))
    bars = plt.bar(['Untrained Accuracy', 'Trained Accuracy'], [untrained_acc, accuracy], color=['maroon', 'limegreen'], edgecolor='gray', width=0.5)
    for i in range(len(bars)):
        height = bars[i].get_height()
        plt.text(x=bars[i].get_x() + bars[i].get_width()/2, y=height+0.01, s=f"{height:.3f}", ha='center')

    plt.ylabel('Accuracy %')
    plt.title('Evaluation of Untrained vs Trained Model')
    plt.show()
    return accuracy