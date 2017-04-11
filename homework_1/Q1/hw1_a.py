# !/usr/bin/python3

import tweepy
import json

def loadKeys(key_file):
    with open(key_file, 'r') as f:
        data = json.load(f)
    return data['api_key'],data['api_secret'],data['token'],data['token_secret']
    pass
def getFollowers(api, root_user, no_of_followers):
    initial_follower=[]
    raw_follower_data = api.followers(root_user,-1)
    for i in range(no_of_followers):
        alist=(raw_follower_data[i].screen_name,root_user)
        print(alist)
        initial_follower.append(alist)
    return initial_follower
    pass
def getSecondaryFollowers(api, followers_list, no_of_followers):
    secondary_follower=[]
    for k in range(no_of_followers):
        slist=getFollowers(api,followers_list[k][0],no_of_followers)
        secondary_follower.append(slist)
    return secondary_follower
    pass


KEY_FILE = 'keys.json'
ROOT_USER = 'PoloChau'
NO_OF_FOLLOWERS = 10
NO_OF_FRIENDS = 10
api_key, api_secret, token, token_secret = loadKeys(KEY_FILE)
auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(token, token_secret)
api = tweepy.API(auth)

primary_followers = getFollowers(api, ROOT_USER, NO_OF_FOLLOWERS)
secondary_followers = getSecondaryFollowers(api, primary_followers, NO_OF_FOLLOWERS)
print(primary_followers[3][0])


