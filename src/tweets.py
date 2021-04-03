import tweepy
import typing as T
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from src.settings import Config


@dataclass
class Tweet:
    id_str: str
    created_at: datetime
    text: str


DATA_DIR = '../'

def get_all_tweets(screen_name, config: Config):
    # Twitter only allows access to a users most recent 3240 tweets with this method

    # authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(config.CONSUMER_KEY, config.CONSUMER_SECRET)
    auth.set_access_token(config.ACCESS_KEY, config.ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name=screen_name, count=200, tweet_mode='extended')

    # save most recent tweets
    alltweets.extend(new_tweets)

    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print(f"getting tweets before {oldest}")

        # all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest, tweet_mode='extended')

        # save most recent tweets
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print(f"...{len(alltweets)} tweets downloaded so far")

    # transform the tweepy tweets into a 2D array that will populate the csv
    outtweets = [Tweet(id_str=tweet.id_str,
                       created_at=tweet.created_at,
                       text=tweet.full_text) for tweet in alltweets]
    return outtweets

def write_tweets(tweets: T.List[Tweet], outpath: Path) -> None:
    pd.DataFrame(tweets).to_csv(outpath)



config = Config()
tweets = get_all_tweets('massdcr', Config())
print(tweets[-1])
outpath = config.DATA_DIR / 'new_massdcr_tweets.csv'
print(outpath)
write_tweets(tweets, outpath)
# df = pd.read_csv('new_massdcr_tweets.csv',parse_dates=['created_at'])
