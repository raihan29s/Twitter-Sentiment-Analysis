# Rating Codes: 1 = negative, 2 = positive, 3 = mixed, 4 = other
### Have 744 Positive and 744 Negative Labels Dataset
"""To divide the dataset debate_assignment.csv into positive and negative labels of equal size 744"""
import csv
# original 4 labelled dataset
file = "data/debate_assignment.csv"

# POSITIVE & NEGATIVE FILES
p1 = "data/positive_data.csv"
n1 = "data/negative_data.csv"

pos_f = open(p1, 'wb')
neg_f = open(n1, 'wb')

reader = csv.reader(open(file, 'r'), delimiter=',')
positive_count = 0
negative_count = 0
tweet_count = 0


for it in reader:
    if len(it) < 2:
        print "length is small", it
        continue
    tweet_count += 1
    tweet = it[1]
    label = int(it[0])

    if positive_count == 744 and negative_count == 744:
        print "BREAKING"
        break
    else:
        if label == 1:  # negative labelled data
            print label
            if negative_count == 744:
                pass
            else:
                negative_count += 1
                neg_f.write(str(label))
                neg_f.write(",")
                neg_f.write(tweet)
                neg_f.write("\n")
        elif label == 2:  # positive labelled data
            print label
            if positive_count == 744:
                pass
            else:
                positive_count += 1
                pos_f.write(str(label))
                pos_f.write(",")
                pos_f.write(tweet)
                pos_f.write("\n")
        print negative_count, positive_count

pos_f.close()
neg_f.close()
