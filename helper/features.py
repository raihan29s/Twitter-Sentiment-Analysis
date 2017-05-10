from helper.tweet_gnome import word_split


def get_train_test_features(features, negdata, posdata):
    negative_features = [(features(f), 'neg') for f in word_split(negdata)]
    positive_features = [(features(f), 'pos') for f in word_split(posdata)]

    negcutoff = len(negative_features)*3/4
    poscutoff = len(positive_features)*3/4

    training_features = negative_features[:negcutoff] + positive_features[:poscutoff]
    testing_features = negative_features[negcutoff:] + positive_features[poscutoff:]
    return training_features, testing_features, positive_features, negative_features
