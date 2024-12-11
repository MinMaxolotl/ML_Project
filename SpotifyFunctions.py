import sys
import spotipy
import json
import numpy as np
import Authenticate

# basically i cant use this entire file because spotify API got destroyed


# Pulls a specified playlist from your Spotify account
# Input:  Spotify Object, Case sensitive string playlist_name, 
# Output: List of items from the detected playlist
def getInputPlaylist(sp, playlist_name):
    playlists = sp.current_user_playlists()['items']
    playlist_ID = None
    for i in range(len(playlists)):
        if playlists[i]['name'] == playlist_name:
            playlist_ID = playlists[i]['id']
            print("Playlist Found!")
            break
        if i == len(playlists) - 1:
            print("Playlist not found :[")
            return -1

    input_playlist = sp.playlist_items(playlist_ID, additional_types="track")['items']

    return input_playlist

# Gets the Spotify IDs for every tack in the input playlist
# Input:  List of playlist items 
# Output: ID of every song in the input
def getPlaylistTrackIDs(input_playlist):
    ID_list = []
    for i in range(len(input_playlist)):
        ID_list.append(input_playlist[i]['track']['id'])

    return np.array(ID_list)

# Gets audio features of every song from the IDs
# Input:  Spotify Object, List of Spotify IDs, Spotify object
# Output: (n_songs, 9) array of song audio features
# No longer works, spotify disabled API features
def getAudioFeatures(sp, ids):
    num_songs = len(ids)
    features = []
    for i in range(num_songs):
        details = sp.audio_features(ids[i])
        # print(details)
        array = [details[0]['acousticness'],
                 details[0]['danceability'],
                 details[0]['energy'],
                 details[0]['instrumentalness'],
                 details[0]['liveness'],
                 details[0]['loudness'],
                 details[0]['speechiness'],
                 details[0]['tempo'],
                 details[0]['valence'],]
        
        features.append(array)
        
    return np.array(features)

# Gets the list of features for a specific playlist
# Input:  The name of a playlist from a user as a string
# Output: Features of every song in the playlist as a dictionary
def name2features(playlist_name):
    spotify = Authenticate.get_spotify()
    input = getInputPlaylist(spotify, playlist_name)
    IDs = getPlaylistTrackIDs(input)
    features = getAudioFeatures(spotify, IDs)
    return IDs, features


# ok i use one spotify function to make a new playlist of the new songs 
def makeSimilarPlaylist(sp, song_ids, playlist_name):
    # first check if playlist name already exists on user profile, if so, delete it
    playlists = sp.user_playlists(sp.current_user()['id'])
    for playlist in playlists['items']:
        if playlist['name'] == playlist_name:
            sp.user_playlist_unfollow(sp.current_user()['id'], playlist['id'])

    # next create a new playlist to the requested name
    similar_playlist = sp.user_playlist_create(sp.current_user()['id'], playlist_name)

    # now add the similar songs to the playlist
    sp.user_playlist_add_tracks(sp.current_user()['id'], similar_playlist['id'], song_ids)

    print(f"New playlist made! It is named: {playlist_name}")


