# Rating Codes: 1 = negative, 2 = positive, 3 = mixed, 4 = other

### Have 744 Positive and 744 Negative Labels Dataset

"""To assign labels to the original dataset"""
import csv
file = "data/debate08_sentiment_tweets.tsv"


of = "data/debate_assignment.csv"
out_f = open(of, 'wb')

reader = csv.reader(open(file, 'rU'), dialect=csv.excel_tab)
c = 0
pos_count = 0
neg_count = 0
for it in reader:
    if len(it) < 5:
        continue
    positive_words = []
    negative_words = []
    # mixed_words = []
    # other_words = []
    single_tweet = it[2]
    ratings = it[5:]
    for rating in ratings:
        if rating == '':
            continue
        rating = int(rating)
        if rating == 1:
            negative_words.append(rating)
        if rating == 2:
            positive_words.append(rating)

    len_po = len(positive_words)
    len_ne = len(negative_words)
    if(len_po > len_ne):
        if(len_po > len_mi):
            if(len_po > len_ot):
                label = 2 # POSITIVE
            else:
                label = 4 # OTHERS
        else:
            if(len_mi > len_ot):
                label = 3 # MIXED
            else:
                label = 4 # OTHERS
    else:
        if(len_ne > len_mi):
            if(len_ne > len_ot):
                label = 1 # NEGATIVE
            else:
                label = 4 # OTHERS
        else:
            label = 3 # MIXED

    out_f.write(str(label))
    out_f.write(",")
    out_f.write(single_tweet)
    out_f.write("\n")
out_f.close()
