import datetime as dt
from pathlib import Path
import re

import tweepy
import attr
from attr.validators import instance_of, optional, deep_iterable
import typing as T
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from src.settings import Config

DEFAULT_HOURS_FROM_OPEN_THRESHOLD = 1.
DATA_DIR = '../'
TZ = 'UTC'


@attr.s
class TweetText:
    text: str = attr.ib(validator=instance_of(str))

    def __attrs_post_init__(self):
        # clean up text
        self.text = self.text.lower().replace('-', '').replace(',', ' ')

    def parse_effective_hour_str(self) -> T.Optional[str]:
        """ return time from tweet

        e.g.
        >parse_effective_time_text(' open at 4:00pm')
        >'4:00pm'
        # TODO improve the regex and match logic
        """
        matches = re.findall('((1[0-2]|0?[1-9]):([0-5][0-9])\s?([AaPp][Mm]))|((1[0-2]|0?[1-9])([AaPp][Mm]))', self.text)
        if len(matches) == 0:
            return None
        else:
            matches = matches[0]
            reopen_time_str = None
            for x in matches:
                if x != '':
                    reopen_time_str = x
                    break
            return reopen_time_str

    @property
    def is_open(self) -> T.Optional[bool]:
        """ Parse text to decide if Pond opening, closing or neither

        :returns: True if opening, False if closing, None if neither

        """
        reopen = False
        close = False

        if 'reopen' in self.text:
            reopen = True
        if 'close' in self.text:
            close = True

        if (reopen and close):
            # scan through and take first occurence
            for i in self.text.split(' '):
                if 'reopen' in i:
                    return True
                elif 'close' in i:
                    return False

        elif reopen:
            return True
        elif close:
            return False
        else:
            return None


def parse_raw_df(df):
    if df.created_at.dt.tz is None:
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize("UTC").dt.tz_convert(TZ)
    return df

@attr.s
class Tweet:
    created_at: pd.Timestamp = attr.ib(validator=instance_of(pd.Timestamp))
    text: TweetText = attr.ib(validator=instance_of(TweetText))
    hours_from_open_threshold: T.Optional[float] = attr.ib(validator=optional(instance_of(float)),
                                                           default=DEFAULT_HOURS_FROM_OPEN_THRESHOLD)

    @classmethod
    def from_dict(cls, d: T.Dict) -> "Tweet":
        return cls(created_at=d['created_at'], text=TweetText(d['text']))

    @property
    def time_from_text(self) -> T.Optional[pd.Timestamp]:
        """ Tries to parse a time from tweet

        :returns: pd.Timestamp if a time was found, else None

        e.g.
        tweet = Tweet(text='opens at 4:00pm',created_at=pd.Timestamp('2020-07-04 3:00pm'))
        time = tweet.time_from_text
        time
        > Timestamp('2020-07-04 16:00:00-0400', tz='EST5EDT')
        """
        reopen_time = self.text.parse_effective_hour_str()
        if reopen_time is not None:
            date_str = self.created_at.date().strftime('%Y-%m-%d')
            reopen_datetime_str = date_str + ' ' + reopen_time
            return pd.Timestamp(ts_input=reopen_datetime_str, tz=TZ)
        else:
            return None

    @property
    def effective_time(self) -> pd.Timestamp:
        tft = self.time_from_text
        # if time exists and is less than 1 hour from created_at time, use it
        if tft and ((tft - self.created_at).total_seconds() / 3600 < self.hours_from_open_threshold):
            return tft
        else:
            return self.created_at

    def to_dict(self) -> T.Dict:
        return {"open": self.text.is_open,
                "effective_time": self.effective_time,
                "created_at": self.created_at,
                "text": self.text.text}

    def to_series(self) -> pd.Series:
        return pd.Series(self.to_dict())

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict(), index=[0])

    def __str__(self) -> str:
        return f"created_at: {self.created_at}\n" \
               f"text: {self.text}"

    def __lt__(self, other: "Tweet") -> bool:
        return self.effective_time < other.effective_time


def num_changes(x):
    return sum(x.iloc[:-1] != x.shift(-1).iloc[:-1])


@attr.s
class Day:
    date: dt.date = attr.ib(validator=instance_of(dt.date))
    open_hour: int = attr.ib(validator=instance_of(int))
    close_hour: int = attr.ib(validator=instance_of(int))
    tweets: T.Sequence[Tweet] = attr.ib(validator=deep_iterable(instance_of(Tweet), instance_of(T.Sequence)))

    def __attrs_post_init__(self):
        assert self.open_hour < self.close_hour, "can't close before opening"
        assert (self.close_hour > 12) and (self.close_hour < 22), f"unreasonable close_hour: {self.close_hour}"
        assert all((t.created_at.date() == self.date) for t in self.tweets), f"not all tweets created_at: {self.date}"
        self.tweets = sorted(self.tweets)

    #         df = self.to_df().iloc[:-1]
    #         changes = num_changes(df['open'])
    #         if  changes!=(len(df)-1):
    #             print(f"detected bad parsing of tweets: \n {changes} \n {self.date} \n {df} "
    #                   " You should run day.remove_inconsistent_tweets")

    @classmethod
    def from_raw_df(cls, df: pd.DataFrame, date: dt.date, open_hour: int, close_hour: int) -> "Day":
        assert set(("created_at", "text")).issubset(set(df.columns))
        df = parse_raw_df(df)
        df = df[df['created_at'].dt.date == date]
        tweets = [Tweet.from_dict(df.iloc[i].to_dict()) for i in range(len(df))]
        return cls(date=date, open_hour=open_hour, close_hour=close_hour, tweets=tweets)

    @property
    def open_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(f"{self.date} {self.open_hour}:00", tz=TZ)

    @property
    def close_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(f"{self.date} {self.close_hour}:00", tz=TZ)

    def remove_inconsistent_tweets(self, inplace: bool = False) -> T.Optional["Day"]:
        # with no dummies, assume park is initially open
        last_open_value = True
        good_tweets = []
        for i, tweet in enumerate(self.tweets):
            if last_open_value == tweet.text.is_open:
                continue
            else:
                good_tweets.append(tweet)
                last_open_value = tweet.text.is_open
        if inplace:
            self.tweets = good_tweets
        else:
            return Day(self.date,
                       open_hour=self.open_hour,
                       close_hour=self.close_hour,
                       tweets=good_tweets)

    def _dummy_open_tweet(self) -> Tweet:
        return Tweet(created_at=self.open_timestamp, text=TweetText('dummy reopen'))

    def _dummy_close_tweet(self) -> Tweet:
        return Tweet(created_at=self.close_timestamp, text=TweetText('dummy closed'))

    def to_df(self, with_dummies: bool = True) -> pd.DataFrame:
        df = self.tweets
        if with_dummies:
            df = [self._dummy_open_tweet()] + df + [self._dummy_close_tweet()]
        df = (pd.concat([x.to_df() for x in df], ignore_index=True))
        df = (df
              .drop_duplicates()
              .dropna()
              .sort_values(by='effective_time'))
        return df

    def to_time_series(self, resample_freq='1H') -> pd.DataFrame:
        df = self.to_df().sort_values(by='effective_time')
        new_index = pd.date_range(start=self.open_timestamp,
                                  end=self.close_timestamp,
                                  freq=resample_freq, tz=TZ)
        df = (df.set_index('effective_time', drop=False)
              .reindex(new_index, method='ffill'))
        df.index.name = 'time'
        df['date'] = self.date
        return df

    def to_time_intervals(self) -> pd.DataFrame:
        df = self.to_df()
        intervals = []
        opens = []
        for i in range(len(df) - 1):
            intervals.append(pd.Interval(df.iloc[i]['effective_time'],
                                         df.iloc[i + 1]['effective_time'],
                                         closed='both'))
            opens.append(df.iloc[i]['open'])
        df = pd.DataFrame({'open': opens, 'intervals': intervals})
        df['length'] = df['intervals'].apply(lambda x: x.length)
        df['date'] = self.date
        return df


@attr.s
class Season:
    days: T.Sequence[Day] = attr.ib(validator=deep_iterable(instance_of(Day), instance_of(T.Sequence)))

    @classmethod
    def from_raw_df(cls, df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp,
                    open_hour: T.Union[T.Sequence[int], int],
                    close_hour: T.Union[T.Sequence[int], int],
                    ) -> "Season":
        assert set(("created_at", "text")).issubset(set(df.columns))
        assert type(open_hour) == type(close_hour)
        date_range = pd.date_range(start_date, end_date, freq='1D')
        ndays = len(date_range)
        if type(open_hour) != int:
            assert len(open_hour) == ndays
            assert len(open_hour) == len(close_hour)
        else:
            open_hour = [open_hour] * ndays
            close_hour = [close_hour] * ndays
        df = parse_raw_df(df)
        df = df[(df['created_at'].dt.date >= start_date) & (df['created_at'].dt.date <= end_date)]
        days = [Day.from_raw_df(df=df, date=d, open_hour=oh, close_hour=ch)
                for d, oh, ch in zip(date_range, open_hour, close_hour)]
        return cls(days=days)

    def remove_inconsistent_tweets(self, inplace: bool = False) -> T.Optional["Season"]:
        # if inplace, this returns nothing, and our job is done
        consistent_days = [d.remove_inconsistent_tweets(inplace=inplace) for d in self.days]
        # if not inplace, we have the good days, now remake a Season
        if not inplace:
            return Season(days=consistent_days)

    def to_df(self):
        return pd.concat([x.to_df() for x in self.days], ignore_index=True)

    def to_time_series(self, resample_freq='1H'):
        return pd.concat([x.to_time_series(resample_freq) for x in self.days])

    def to_time_intervals(self):
        return pd.concat([x.to_time_intervals() for x in self.days])


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
    outtweets = [Tweet(created_at=tweet.created_at,
                       text=tweet.full_text) for tweet in alltweets]
    return outtweets


def write_tweets(tweets: T.List[Tweet], outpath: Path) -> None:
    pd.DataFrame([t.__dict__ for t in tweets]).to_csv(outpath, index=False)


def test():

    config = Config()
    user = 'waldenpondstate'
    outpath = config.DATA_DIR / f'{user}.csv'
    # start_dt = pd.Timestamp('2020-05-01')
    # end_dt = pd.Timestamp('2020-10-01')
    start_dt = pd.Timestamp('2020-05-01', tz=TZ)
    end_dt = pd.Timestamp('2020-10-01', tz=TZ)
    resample_freq = '15T'
    open_hour = 7
    close_hour = 20

    # tweets = get_all_tweets(user, Config())
    # print(tweets[-1])
    # print(outpath)
    # write_tweets(tweets, outpath)

    df = pd.read_csv(outpath, parse_dates=['created_at'])
    season = Season.from_raw_df(df,
                                start_date=start_dt,
                                end_date=end_dt,
                                open_hour=open_hour,
                                close_hour=close_hour,
                                )
    season = season.remove_inconsistent_tweets()

    sts = season.to_time_series(resample_freq=resample_freq)
    print(sts.tail())
    sti = season.to_time_intervals()
    print(sti.tail())


if __name__ == "__main__":
    test()
