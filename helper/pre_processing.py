import re
from util.stopwords import stopwords

def process_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    return tweet

def replace_two_or_more(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


def filter_words(words):
    w = replace_two_or_more(words)
    w = w.strip('\'"?,.')
    val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
    if(w in stopwords or val is None):
        pass
    else:
        return w.lower()