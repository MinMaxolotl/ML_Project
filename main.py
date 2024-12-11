import keras
from Authenticate import Authenticate
from NueralNetwork import makeFFNN, trainFFNN, inference, modelEvaluate
from Database import getDatasetFeatures, findSimilarSongs
from SpotifyFunctions import makeSimilarPlaylist
import numpy as np

spotify, database = Authenticate()
training_data = getDatasetFeatures(database, 'Training Set')
testing_data = getDatasetFeatures(database, 'Testing Set')
playlist1 = getDatasetFeatures(database, 'Playlist1')
playlist2 = getDatasetFeatures(database, 'Playlist2')
playlist3 = getDatasetFeatures(database, 'Playlist3')

# print(playlist1)

# Feed Forward Neural Network needs defined feautre size, we double it for feature pairs
num_features = playlist2.shape[1]*2

# model = makeFFNN(num_features)
# trained_model = trainFFNN(model, training_data, playlist2, 100)

# top_songs_index = inference(trained_model, testing_data, playlist2, 5)

# song_ids = findSimilarSongs(spotify, database, top_songs_index)
# makeSimilarPlaylist(spotify, song_ids, "SimilarSongs")

# Use these lines for testing trained model
trained_model = keras.models.load_model('trained_model.h5')
modelEvaluate(trained_model, testing_data, playlist2)

# Code for testing all three playlists


# Playlist 1
# model1 = makeFFNN(num_features)
# trained_model1 = trainFFNN(model1, training_data, playlist1, 10)
# acc1 = modelEvaluate(trained_model1, testing_data, playlist1)

# # Playlist 2
# model2 = makeFFNN(num_features)
# trained_model2 = trainFFNN(model2, training_data, playlist2, 10)
# acc2 = modelEvaluate(trained_model2, testing_data, playlist2)

# # Playlist 3
# model3 = makeFFNN(num_features)
# trained_model3 = trainFFNN(model3, training_data, playlist3, 10)
# acc3 = modelEvaluate(trained_model3, testing_data, playlist3)

# print(f"Average test accuracy: {np.mean([acc1, acc2, acc3])}")