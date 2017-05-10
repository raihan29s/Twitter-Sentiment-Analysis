import csv
from helper.pre_processing import process_tweet

def word_split(data):
    data_new = []
    for word in data:
        word_filter = [i.lower() for i in word.split()]
        data_new.append(word_filter)
    return data_new

def word_feats(words):
    return dict([(word, True) for word in words])


def get_positive_dataset():
    with open('data/positive_data.csv', 'rb') as f:
        positives = []
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            positives.append(process_tweet(row[1]))
    return positives


def get_negative_dataset():
    with open('data/negative_data.csv', 'rb') as f:
        negatives = []
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            negatives.append(process_tweet(row[1]))
    return negatives


