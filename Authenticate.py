import spotipy
import certifi
from pymongo import MongoClient
from spotipy.oauth2 import SpotifyOAuth

#Authenticate Spotify API Connection
def get_spotify():
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="#",
                                               client_secret="#",
                                               redirect_uri="http://localhost:1234",
                                               scope=["user-library-read", "playlist-modify-public", "playlist-modify-private"]))

    return sp

#Authenticate Database Connection to Spotify Songs dataset. From https://www.mongodb.com/resources/languages/python#querying-in-python
def get_collection():
 
   # Provide the mongodb atlas url to connect python to mongodb using pymongo
   CONNECTION_STRING = "#"
 
   # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
   client = MongoClient(CONNECTION_STRING, maxPoolSize=10, maxIdleTimeMS=60000, tlsCAFile=certifi.where())

   # Create the database for our example (we will use the same database throughout the tutorial
   return client['songs']

def Authenticate():
    spotify = get_spotify()
    database = get_collection()

    return spotify, database
