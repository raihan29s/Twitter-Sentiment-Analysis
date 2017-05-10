def printing_results(accuracy=None, precision=None, recall=None, f_measure=None, \
                     fold=None, classifier_name=None):
    print 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    print  "(", classifier_name, ")", ' %d FOLD RESULT ' % fold, "results"
    print '---------------------------------------'
    print 'Accuracy:', accuracy
    print 'Precision:', precision
    print 'Recall:', recall
    print 'F-measure:', f_measure
    print 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
