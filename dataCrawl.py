import subredditsConfig
import APIconfig
import praw
import json
import sys
import os

reddit = praw.Reddit(client_id=APIconfig.CLIENT_ID, client_secret=APIconfig.CLIENT_SECRET,
                     password=APIconfig.PASSWORD, user_agent=APIconfig.USER_AGENT,
                     username=APIconfig.USERNAME)



def write(subreddit, counter, text):
    if (counter <= (subredditsConfig.POSTCOUNT * .25)):
        directory = 'test/' + subreddit + '/'
    elif (counter > (subredditsConfig.POSTCOUNT * .25) and counter <= (subredditsConfig.POSTCOUNT * .5)):
        directory = 'dev/' + subreddit + '/'
    else:
        directory = 'train/' + subreddit + '/'

    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + str(counter) + '.txt', 'w') as outfile:
        json.dump(text, outfile)

counter = 0
for submission in reddit.subreddit(subredditsConfig.SUBREDDIT1).hot(limit=subredditsConfig.POSTCOUNT):
    write(subredditsConfig.SUBREDDIT1, counter, submission.selftext)
    counter +=1

counter = 0
for submission in reddit.subreddit(subredditsConfig.SUBREDDIT2).hot(limit=subredditsConfig.POSTCOUNT):
    write(subredditsConfig.SUBREDDIT2, counter, submission.selftext)
    counter +=1
