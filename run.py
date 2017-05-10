import random
import collections
import nltk.classify.util
from nltk.metrics.scores import precision, accuracy, recall, f_measure, log_likelihood, approxrand
from nltk.classify import NaiveBayesClassifier, MaxentClassifier, SklearnClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import *
from util.stopwords import stopwords
from util.classifier_list import classifier_list
from helper.tweet_gnome import get_negative_dataset, get_positive_dataset, word_feats
from helper.features import get_train_test_features
from helper.printer import printing_results
from helper.writer import write_to_file


def evaluate_classifier_performance(features, posdata, negdata):

    training_features, testing_features, positive_features, negative_features = get_train_test_features(features, posdata, negdata)

    for cl in classifier_list:
        f = open('results/data/' + cl + '1.txt', 'w')
        if cl == 'maxent':
            classifier_name = 'Maximum Entropy Classifier'
            classifier = MaxentClassifier.train(training_features, 'GIS', trace=0, encoding=None, labels=None, gaussian_prior_sigma=0, max_iter=1)
        elif cl == 'svm':
            classifier_name = 'SVM Classifier'
            classifier = SklearnClassifier(LinearSVC(), sparse=False)
            classifier.train(training_features)
        else:
            classifier_name = 'Naive Bayes Classifier'
            classifier = NaiveBayesClassifier.train(training_features)

        reference_sets = collections.defaultdict(set)
        test_sets = collections.defaultdict(set)

        for i, (feats, label) in enumerate(testing_features):
            reference_sets[label].add(i)
            observed = classifier.classify(feats)
            test_sets[observed].add(i)
        
        accuracy_m = nltk.classify.util.accuracy(classifier, testing_features)
        pos_precision = precision(reference_sets['pos'], test_sets['pos'])
        pos_recall = recall(reference_sets['pos'], test_sets['pos'])
        pos_fmeasure = f_measure(reference_sets['pos'], test_sets['pos'])
        neg_precision = precision(reference_sets['neg'], test_sets['neg'])
        neg_recall = recall(reference_sets['neg'], test_sets['neg'])
        neg_fmeasure = f_measure(reference_sets['neg'], test_sets['neg'])

        precision_m = (pos_precision + neg_precision) / 2
        recall_m = (pos_recall + neg_recall) / 2
        f_measure_m = (pos_fmeasure + neg_fmeasure) / 2
        write_to_file(f, accuracy_m, precision_m, recall_m, f_measure_m)
        printing_results(accuracy=accuracy_m, precision=precision_m, recall=recall_m, f_measure=f_measure_m, fold=1, classifier_name=classifier_name)
    training_features = negative_features + positive_features
    return training_features


def k_fold_cross_validation(training_features, fold=5):
    random.shuffle(training_features)

    for cl in classifier_list:
        f = open('results/data/' + cl + '5.txt', 'w')
        subset_size = len(training_features) / fold
        accuracy = []
        pos_precision = []
        pos_recall = []
        neg_precision = []
        neg_recall = []
        pos_fmeasure = []
        neg_fmeasure = []
        cv_count = 1
        for i in range(fold):
            testing_this_round = training_features[i*subset_size:][:subset_size]
            training_this_round = training_features[:i*subset_size] + training_features[(i+1)*subset_size:]

            if cl == 'maxent':
                classifier_name = 'Maximum Entropy Classifier'
                classifier = MaxentClassifier.train(training_this_round, 'GIS', trace=0, encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 1)
            elif cl == 'svm':
                classifier_name = 'SVM Classifier'
                classifier = SklearnClassifier(LinearSVC(), sparse=False)
                classifier.train(training_this_round)
            else:
                classifier_name = 'Naive Bayes Classifier'
                classifier = NaiveBayesClassifier.train(training_this_round)

            reference_sets = collections.defaultdict(set)
            test_sets = collections.defaultdict(set)
            for i, (feats, label) in enumerate(testing_this_round):
                reference_sets[label].add(i)
                observed = classifier.classify(feats)
                test_sets[observed].add(i)

            cv_accuracy = nltk.classify.util.accuracy(classifier, testing_this_round)
            cv_pos_precision = precision(reference_sets['pos'], test_sets['pos'])
            cv_pos_recall = recall(reference_sets['pos'], test_sets['pos'])
            cv_pos_fmeasure = f_measure(reference_sets['pos'], test_sets['pos'])
            cv_neg_precision = precision(reference_sets['neg'], test_sets['neg'])
            cv_neg_recall = recall(reference_sets['neg'], test_sets['neg'])
            cv_neg_fmeasure = f_measure(reference_sets['neg'], test_sets['neg'])

            accuracy.append(cv_accuracy)
            pos_precision.append(cv_pos_precision)
            pos_recall.append(cv_pos_recall)
            neg_precision.append(cv_neg_precision)
            neg_recall.append(cv_neg_recall)
            pos_fmeasure.append(cv_pos_fmeasure)
            neg_fmeasure.append(cv_neg_fmeasure)

            cv_count += 1
            accuracy_m = sum(accuracy)/fold
            precision_m = (sum(pos_precision)/fold + sum(neg_precision)/fold) / 2
            recall_m = (sum(pos_recall)/fold + sum(neg_recall)/fold) / 2
            fmeasure_m = (sum(pos_fmeasure)/fold + sum(neg_fmeasure)/fold) / 2

        write_to_file(f, accuracy_m, precision_m, recall_m, fmeasure_m)
        printing_results(accuracy=accuracy_m, precision=precision_m, \
                             recall=recall_m, f_measure=fmeasure_m, fold=fold,\
                             classifier_name=classifier_name)


if __name__ == "__main__":
    posdata = get_positive_dataset()
    negdata = get_negative_dataset()
    stopset = set(stopwords) - set(('over', 'under', 'below', 'more', 'most', 'no', 'not', 'only', 'such', 'few', 'so', 'too', 'very', 'just', 'any', 'once'))
    training_f = evaluate_classifier_performance(word_feats, negdata, posdata)
    k_fold_cross_validation(training_f)

