import os
import sys
from tweepy import API
from tweepy import OAuthHandler



"""Setup Twitter authentication.
Return: tweepy.OAuthHandler object
"""

def get_twitter_auth():
    try:
        consumer_key = '0NfwllYwUfHydecGf6pHIfIa9'
        consumer_secret = 'LVZUqFBT9XOxqRsTHVlcqFz1UtzXIN3a2SYOuCdegjN9OYifnV'
        access_token = '86742546-zfcU9tgKracUz5j0zQH3gW4wht4AgMmt1fbgV9RSl'
        access_secret = 'Eb60sdlg7dPGsW0QZJ0eENSMUNAOoSIoyZWRRWKMAkhqf'
    except KeyError:
        sys.stderr.write("TWITTER_* environment variables not set\n")
        sys.exit(1)
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return auth


"""Setup Twitter API client.
Return: tweepy.API object"""

def get_twitter_client():
    auth = get_twitter_auth()
    client = API(auth)
    return client

