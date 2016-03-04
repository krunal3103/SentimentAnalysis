import sys
import collections
from sklearn.naive_bayes import BernoulliNB
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
import warnings
warnings.filterwarnings("ignore")
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)

    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg

def removeStopWords(stopwords,dataset):
  d=dict()
  for row in dataset:
    #removing duplicates
    row=set(row)
    for word in row:
      #removing stopwords
      if word not in stopwords:
        if word in d:
          d[word]=d[word]+1
        else:
          d[word]=1
  return d


def part3(pos,neg,m_pos,m_neg):
  temp=list()
  for k in pos:
    if k in neg:
      if (pos[k]/neg[k]>=2) and (pos[k]>=m_pos or neg[k]>=m_neg):
        temp.append(k)
        del neg[k]
    else:
      try:
        if pos[k]>=m_pos or neg[k]>=m_neg:
          temp.append(k)
      except:
        pass
  for k in neg:
    if k in pos:
      if neg[k]/pos[k]>=2  and (pos[k]>=m_pos or neg[k]>=m_neg):
        temp.append(k)
        del pos[k]
    else:
      try:
        if pos[k]>=m_pos or neg[k]>=m_neg:
          temp.append(k)
      except:
        pass
  temp=list(set(temp))
  return temp

def getFeatureList(features,dataset):
  temp=list()
  for row in dataset:
    d=dict()
    for word in features:
      if word in row:
        d[word]=1
      else:
        d[word]=0
    temp.append(list(d.values()))
  return temp
def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))


    # Determine a list of words that will be used as features.
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.

    #get for train_pos
    d_train_pos=removeStopWords(stopwords,train_pos)
    #print len(d_train_pos)
    #get for train_pos
    d_train_neg=removeStopWords(stopwords,train_neg)
    #print len(d_train_neg)
    m_pos=0.01*len(train_pos)
    m_neg=0.01*len(train_neg)
    feature_list=part3(d_train_pos,d_train_neg,m_pos,m_neg)
    #print len(feature_list)

    train_pos_vec=getFeatureList(feature_list,train_pos)
    #print train_pos_vec
    train_neg_vec=getFeatureList(feature_list,train_neg)
    #print train_pos_vec
    test_pos_vec=getFeatureList(feature_list,test_pos)

    test_neg_vec=getFeatureList(feature_list,test_neg)

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    l_train_pos = []
    l_train_neg = []
    l_test_pos = []
    l_test_neg = []

    for i in range(len(train_pos)):
        label1 = 'TRAIN_POS_'+str(i)
        l_train_pos.append(LabeledSentence(words = train_pos[i],tags=[label1]))
    for i in range(len(train_neg)):
        label2 = 'TRAIN_NEG_'+str(i)
        l_train_neg.append(LabeledSentence(words = train_neg[i],tags=[label2]))
    for i in range(len(test_pos)):
        label3 = 'TEST_POS_'+str(i)
        l_test_pos.append(LabeledSentence(words = test_pos[i],tags=[label3]))
    for i in range(len(test_neg)):
        label4 = 'TEST_NEG_'+str(i)
        l_test_neg.append(LabeledSentence(words = test_neg[i],tags=[label4]))

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = l_train_pos + l_train_neg + l_test_pos + l_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []
    for i in range(len(train_pos)):
        label1 = 'TRAIN_POS_'+str(i)
        train_pos_vec.append(model.docvecs[label1])

    for i in range(len(train_neg)):
        label2 = 'TRAIN_NEG_'+str(i)
        train_neg_vec.append(model.docvecs[label2])

    for i in range(len(test_pos)):
        label3 = 'TEST_POS_'+str(i)
        test_pos_vec.append(model.docvecs[label3])

    for i in range(len(test_neg)):
        label4 = 'TEST_NEG_'+str(i)
        test_neg_vec.append(model.docvecs[label4])
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LogisticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    X = train_pos_vec + train_neg_vec
    nb_model = BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit(X,Y)

    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(X,Y)
    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters

    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X = train_pos_vec + train_neg_vec
    nb_model = sklearn.naive_bayes.GaussianNB()
    nb_model.fit(X,Y)

    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(X,Y)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters

    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    for x in test_pos_vec:
        if model.predict(x) == ['pos']:
            tp = tp + 1
        else:
            fn = fn + 1

    for x in test_neg_vec:
        if model.predict(x) == ['neg']:
            tn = tn + 1
        else:
            fp = fp + 1
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
