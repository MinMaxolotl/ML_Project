from pymongo import MongoClient
import sys
import spotipy
import json
import numpy as np
from Authenticate import Authenticate
from SpotifyFunctions import makeSimilarPlaylist


# Function to pull features from a song collection and order them into an array of arrays
def getDatasetFeatures(database, collection_name):
    # call training dataset from database
    training_songs = database[collection_name]

    # parse through mongoDB database and grab all of the audio features
    song_features = training_songs.find({}, {
                                                "acousticness": 1,
                                                "danceability": 1,
                                                "energy": 1,
                                                "instrumentalness": 1,
                                                "liveness": 1,
                                                "loudness": 1,
                                                "speechiness": 1,
                                                "tempo": 1,
                                                "valence": 1,
                                                "_id": 0                # excluding the id field, unnecessary
                                            })

    feature_array = []
   
    # convert to an array of arrays. This is the feauture array for all the datasets
    for song in song_features:
        feature_array.append([   
                            song.get('acousticness', 0),
                            song.get('danceability', 0),
                            song.get('energy', 0),
                            song.get('instrumentalness', 0),
                            song.get('liveness', 0),
                            song.get('loudness', 0),
                            song.get('speechiness', 0),
                            song.get('tempo', 0),
                            song.get('valence', 0)
                            ])
    
    return np.array(feature_array)

# Testing getDatasetFeatures(). It works!
# playlist1 = getDatasetFeatures('Playlist1')

# Function that returns the name and artist of most similar songs in testing dataset from index
def findSimilarSongs(sp, database, top_songs_index):
    # Gather testing dataset information
    testing_songs = database['Testing Set']

    # Create list of song data and list of song ids
    similar_songs = []
    ids = []

    # For every index in the index list, recover the name and title of the song.
    for index in top_songs_index:
        # call all the songs, skip to the index, and only get the value at that index
        song = testing_songs.find().skip(int(index)).limit(1)
        song_list = list(song)
        # append both the song title and artist to the list
        title = song_list[0]['song']
        artist = song_list[0]['artist']
        similar_songs.append([title, artist])

    # gather spotify id's for every song
    for song in similar_songs:
        # using the song name and artist, we search for the top result in spotify api to find that song
        search = sp.search(q=f"track:{song[0]} artist:{song[1]}", type='track', limit=1)
        # we use the search data to now get the spotify id
        # if there was no good search, then do not append any ids
        if search['tracks']['items']:
            # if found append the id
            ids.append(search['tracks']['items'][0]['id'])
        else:
            # if no id was found, warn user
            print(f"Warning: The song ID for '{song[0]}' by '{song[1]}' could not be located")
            ids.append(None)
    
    print(similar_songs)
    print(ids)

    return np.array(ids)

# Testing findSimilarSongs(). It works!
# indexes = [0, 3, 6]
# sp, database = Authenticate()
# song_ids = findSimilarSongs(sp, database, indexes)

# Testing makeSimilarPlaylist() with results from above. IT WORKS RAHHH
# makeSimilarPlaylist(sp, song_ids, "SimilarSongs")


# Doesnt work anymore

def dataset2features(sp, ids):
    num_songs = len(ids)
    features = []
    for i in range(num_songs):
        details = sp.audio_features(ids[i])
        # print(details)
        dict = {'id': str(ids[i]),
                'acousticness': details[0]['acousticness'],
                'danceability': details[0]['danceability'],
                'energy': details[0]['energy'],
                'instrumentalness': details[0]['instrumentalness'],
                'liveness': details[0]['liveness'],
                'loudness': details[0]['loudness'],
                'speechiness': details[0]['speechiness'],
                'tempo': details[0]['tempo'],
                'valence': details[0]['valence']}
        features.append(dict)
  
    return features