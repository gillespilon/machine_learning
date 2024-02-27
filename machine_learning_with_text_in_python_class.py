#! /usr/bin/env python3
"""
Essential code for Kevin Markham's "Machine learning with text in PYthon."
"""

from pathlib import Path
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_iris
from sklearn import metrics
import datasense as ds
import pandas as pd


def tokenize_test(vect, X_train, X_test, y_train, y_test) -> None:
    """
    machine_learning_with_text_in_python_class.py
    A function that accepts a CountVectorizer object and calculates accuracy

    Tuning CountVectorizer is a form of feature engineering, the process
    through which you create features that do not natively exist in the data
    set. Your goal is to create features that contain the signal from the data
    with respect to the response value rather than the noise.

    Parameters
    ----------
    vect : sklearn.feature_extraction.text.CountVectorizer
    X_train : pd.Series
    X_test : pd.Series
    y_train : pd.Series
    y_testn : pd.Series
    """
    # create document term matrices DTMs
    X_train_dtm = vect.fit_transform(raw_documents=X_train)
    X_test_dtm = vect.transform(raw_documents=X_test)
    # print the number of features that were generated (number of columns)
    print("Features: ", X_train_dtm.shape[1])
    # use multinomial naive Bayes to predict the star rating
    nb = MultinomialNB()
    nb.fit(X=X_train_dtm, y=y_train)
    y_pred_class = nb.predict(X=X_test_dtm)
    # print the accuracy of its predictions
    print(
        "Accuracy: ",
        metrics.accuracy_score(y_true=y_test, y_pred=y_pred_class),
    )


def main():
    # Week 1 class, working with text data in scikit-learn
    # Model building in scikit-learn (refresher)
    # load the iris data as an example
    iris = load_iris()
    # store the feature matrix X and the response vector y
    X = iris.data
    y = iris.target
    # check the shapes of X and y
    print("shape of X")
    print(X.shape)
    print("shape of y")
    print(y.shape)
    print("X object type")
    print(type(X))
    print("y object type")
    print(type(y))
    # examine the first five rows of X
    print("First five rows of X:")
    print(pd.DataFrame(data=X, columns=iris.feature_names).head())
    # the features must be numeric
    # instantiate the model with the default parameters
    knn = KNeighborsClassifier()
    # fit the model with the data; occurs in-place
    knn.fit(X=X, y=y)
    # predict the response for a new observation
    print("prediction for 3, 5, 4, 2")
    print(knn.predict(X=[[3, 5, 4, 2]]))
    # Representing text as numerical data
    # example text for numerical training (SMS messages)
    simple_train = [
        "call you tonight",
        "Call me a cab",
        "please call me... PLEASE!",
    ]
    # instantiate CountVectorizer() with the default parameters
    vect = CountVectorizer()
    # learn the vocabulary of the training data; occurs in place
    vect.fit(raw_documents=simple_train)
    # examine the fitted vocabulary
    print("Fitted vocabulary")
    print(vect.get_feature_names_out())
    # transform training data into a document term matrix
    simple_train_dtm = vect.transform(raw_documents=simple_train)
    print("DTM sparse matrix")
    print(simple_train_dtm)
    # convert sparse matrix to dense matrix; does not change simple_train_dtm
    # the toarray() method is only so that we can "see" the result
    simple_train_dtm.toarray()
    print("Dense matrix")
    print(simple_train_dtm.toarray())
    print(
        pd.DataFrame(
            data=simple_train_dtm.toarray(),
            columns=vect.get_feature_names_out(),
        )
    )
    # example text for model testing
    simple_test = ["please don't call me"]
    # in order to make a prediction, the new observation must have the same
    # features as the training observations, both in number and meaning
    # transform testing data into a DTM sparse matrix using existing vocabulary
    simple_test_dtm = vect.transform(raw_documents=simple_test)
    # summary
    # vect.fit(train) learns the vocabulary of the training data
    # vect.transform(train) uses the fitted vocabulary to build a DTM
    # from the training data
    # vect.transform (test) uses the fitted voacbulary to build a DTM
    # from the testing data and ignores tokens it had not seen before
    # Reading the SMS data
    # read file into pandas from a URL
    path = Path(
        "machine_learning_with_text_in_python/ML-text-main/data/sms.tsv"
    )
    sms = pd.read_table(
        filepath_or_buffer=path, header=None, names=["label", "message"]
    )
    print("shape of sms")
    print(sms.shape)
    print("look at first ten rows")
    print(sms.head(10))
    print("Examine the class distribution")
    print(sms["label"].value_counts())
    # convert label to a numeric variable
    sms["label_num"] = sms["label"].replace(
        to_replace=["ham", "spam"], value=[0, 1]
    )
    # usual way to define X and y for use with a model
    # X = iris.data
    # y = iris.target
    # required way to define X and y with a CountVectorizer
    # X and y are one-dimensional
    # X is defined as one-dimensional because it will get converted by
    # CountVectorizer into a two-dimensional X
    X = sms["message"]
    y = sms["label_num"]
    # split X and y into training and test data sets
    # this must be done before vectorization
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, test_size=0.25
    )
    print("X_train shape")
    print(X_train.shape)
    print("X_test shape")
    print(X_test.shape)
    # Vectorizing the SMS data
    # instantiate the vectorizer
    vect = CountVectorizer()
    # learn training data vocabulary, then use it to create a sparse DTM
    # the following two lines are as above
    # vect.fit(raw_documents=X_train)
    # X_train_dtm = vect.transform(raw_documents=X_train)
    # alternatively you can do this in one line
    X_train_dtm = vect.fit_transform(raw_documents=X_train)
    # you do not use a fit step on the test data, only on the training data
    # transform test data using fitted vocabulary into a sparse matrix DTM
    X_test_dtm = vect.transform(raw_documents=X_test)
    # Building a Naive Bayes model
    # instantiate a Multinomial Naive Bayes model
    nb = MultinomialNB()
    # train the model using X_train_dtm
    nb.fit(X=X_train_dtm, y=y_train)
    # make class predictions for X_test_dtm (it's a classifcation problem)
    y_pred_class = nb.predict(X=X_test_dtm)
    # calculate accuracy of class predictions
    print("Accuracy of class predictions")
    print(metrics.accuracy_score(y_true=y_test, y_pred=y_pred_class))
    # print the confusion matrix
    # will give you two rows, two columns
    #
    #                     predicted negative (spaM  predicted positive (ham)
    # actual negative (spam)     count of TN             count of FP
    # actual positive (ham)      count of FN             count of TP
    #
    # false positive means message incorrectly classified as spam
    # false negative means message incorrectly classified as ham
    #
    #                    predicted not spam          predicted spam
    # actually not spam  correct predict not spam    incorrect predict spam
    # actually spam      incorrect predict not spam  correct predict spam
    #
    print("Confusion matrix")
    # y_true is for the actual y class labels in the test data
    # y_predicted is for the class labels predicted by the classification model
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_class))
    # print message text for false positives
    print("message text for false positives")
    print(X_test[y_test < y_pred_class])
    # print message text for false negatives
    print("message text for false negatives")
    print(X_test[y_test > y_pred_class])
    print("What do you notice above the false negatives? (e.g. cell 3132)")
    print(X_test[3132])
    print(
        "It may be that there are a lot of 'ham' words in a longer message"
        "that tries to overwhelm the 'spam' words"
    )
    # calculate predicted probabilities for X_test_dtm (poorly calibrated)
    # these probabilities are not very useful
    # class 1 is spam
    print("Calculate predicted probabilities for X_test_dtm classes 0, 1")
    y_pred_prob = nb.predict_proba(X=X_test_dtm)
    print(y_pred_prob)
    print("Calculate predicted probabilities for X_test_dtm class 1")
    y_pred_prob = nb.predict_proba(X=X_test_dtm)[:, 1]
    print(y_pred_prob)
    # calculate AUC (area under the curve)
    print("Calculate AUC (area under the curve)")
    print(metrics.roc_auc_score(y_true=y_test, y_score=y_pred_prob))
    # Comparing Naive Bayes with logistic regression
    print("Comparing Naive Bayes with logistic regression")
    # instantiate a logistic regression model
    logreg = LogisticRegression(n_jobs=-1)
    # train the model using x_train_dtm
    logreg.fit(X=X_train_dtm, y=y_train)
    # make class predictions for X_test_dtm
    y_pred_class = logreg.predict(X=X_test_dtm)
    # calculate predicted probabilities for X_test_dtm (well calibrated)
    # class 1 is spam
    print("Calculate predicted probabilities for X_test_dtm class 1")
    y_pred_prob = logreg.predict_proba(X=X_test_dtm)[:, 1]
    print(y_pred_prob)
    # calculate accuracy
    print("Calculate accuracy")
    print(metrics.accuracy_score(y_true=y_test, y_pred=y_pred_class))
    print("Calculate AUC (area under the curve)")
    print(metrics.roc_auc_score(y_true=y_test, y_score=y_pred_prob))
    # Calculate the "spamminess" of each token
    # Why did some messages get flagged as spam instead of ham?
    # This code (for next many lines) is to diagnose spammy words
    print("Calculate the 'spamminess' of each token")
    # store the vocabulary of X_train
    X_train_tokens = vect.get_feature_names_out()
    print("Vocabulary size")
    print(len(X_train_tokens))
    # naive Bayes counts the number of times each token appears in each class
    print(
        "naive Bayes counts the number of times each token appears "
        "in each class"
    )
    print(nb.feature_count_)
    print(
        "Rows represent classes, columns represent tokens. "
        "Class 0 (first row) represents ham. "
        "Class 1 (second row) represents spam."
    )
    print(nb.feature_count_.shape)
    # number of times each token appears across all ham messages
    print("number of times each token appears across all ham messages")
    ham_token_count = nb.feature_count_[0, :]
    print(ham_token_count)
    # number of times each token appears across all spam messages
    print("number of times each token appears across all spam messages")
    spam_token_count = nb.feature_count_[1, :]
    print(spam_token_count)
    # create a DataFrame of tokens with their separate ham and spam counts
    print(
        "create a DataFrame of tokens with their separate ham and spam counts"
    )
    tokens = pd.DataFrame(
        data={
            "token": X_train_tokens,
            "ham": ham_token_count,
            "spam": spam_token_count,
        }
    ).set_index(keys="token")
    print("Examine five random rows from this DataFrame")
    print(tokens.sample(n=5, random_state=6))
    # Before we can use this to calculate the spamminess of each token,
    # we need to avoid dividing by zero and account for the class imbalance.
    # Add 1 to ham and spam counts to avoid dividing by 0.
    print("Add 1 to ham, spam counts to avoid dividing by 0")
    tokens["ham"] = tokens["ham"] + 1
    tokens["spam"] = tokens["spam"] + 1
    print(tokens.sample(n=5, random_state=6))
    # Naive Bayes counts the number of observations in each class
    # class_count_ is an attribute not a method because it does not have ()
    # the _ at the end is an scikit-learn convention that says this attriute
    # does not exist until you have run a fit
    nb.class_count_
    # convert the ham and spam counts into frequencies
    tokens["ham"] = tokens["ham"] / nb.class_count_[0]
    tokens["spam"] = tokens["spam"] / nb.class_count_[1]
    print("Convert the ham and spam counts into frequencies")
    print(tokens.sample(n=5, random_state=6))
    # calculate the ratio of spam-to-ham for each token
    tokens["spam_ratio"] = tokens["spam"] / tokens["ham"]
    print(tokens.sample(n=5, random_state=6))
    # examine the DataFrame sorted by spam_ratio
    tokens.sort_values(by="spam_ratio", ascending=False)
    print(tokens.sort_values(by="spam_ratio", ascending=False))
    # look up the spam_ratio for a given token
    tokens.loc["dating", "spam_ratio"]
    print("Spam ratio for the 'dating' token")
    print(tokens.loc["dating", "spam_ratio"])
    # In essence, Naive Bayes learns the spam_ratio column to make predictions
    # Creating a DataFrame from individual text files
    # create a list of ham file names within a directory
    directory = Path(
        "machine_learning_with_text_in_python/ML-text-main/data/ham_files"
    )
    ham_file_names = ds.list_files(
        directory=directory, pattern_extension=[".txt"]
    )
    print("List of ham text files within a directory")
    print(ham_file_names)
    # the next three lines is what Kevin used
    # import glob
    # ham_file_names = glob.glob("machine_learning_with_text_in_python/ML-text-main/data/ham_files/*.txt")
    # print(ham_file_names)
    # read the contents of the ham files into a list
    ham_text = []
    for filename in ham_file_names:
        with open(filename) as f:
            ham_text.append(f.read())
    print(ham_text)
    # create a list of spam file names within a directory
    directory = Path(
        "machine_learning_with_text_in_python/ML-text-main/data/spam_files"
    )
    spam_file_names = ds.list_files(
        directory=directory, pattern_extension=[".txt"]
    )
    print("List of spam text files within a directory")
    print(spam_file_names)
    # read the contents of the spam files into a list
    spam_text = []
    for filename in spam_file_names:
        with open(filename) as f:
            spam_text.append(f.read())
    print(spam_text)
    # combine the ham and spam lists
    all_text = ham_text + spam_text
    print(all_text)
    # create a list of labels (ham=0, spam=1)
    # if you have a list with one element in it and you multiply it by a
    # number, it makes a list of that length with the number repeated
    all_labels = [0] * len(ham_text) + [1] * len(spam_text)
    print(all_labels)
    # convert the  lists into a DataFrame
    # CountVectorize will ignore new line code
    df = pd.DataFrame(data={"labels": all_labels, "message": all_text})
    print(df)
    # Week 2 class, basic natural language processing NLP
    print("Basic natural language processing")
    # read yelp.csv into a DataFrame
    print("Read yelp.csv and print the first five rows, and row 0")
    yelp_path = Path(
        "machine_learning_with_text_in_python/ML-text-main/data/yelp.csv"
    )
    yelp = ds.read_file(file_name=yelp_path, encoding="latin-1")
    # print the first five rows
    print(yelp.head())
    # examine the text for the first row
    print(yelp.loc[0, "text"])
    print(
        "Our goal is to distinguish between one star and five star reviews "
        "using only the review text."
    )
    # binary classification is easier
    # you might get better predictions by training your model only on
    # one start and five star reviews. one star and five star reviews have
    # the most polarized language
    print("Examine the class distribution.")
    print(yelp["stars"].value_counts().sort_index())
    print("Create a DataFrame with five star and one star reviews only.")
    # create a new DataFrame that only contains the five star and one star
    # reviews
    # yelp_best_worst = yelp[(yelp["stars"] == 5) | (yelp["stars"] == 1)]
    yelp_best_worse = yelp_raw[yelp_raw["stars"].isin([1, 5])]
    print(yelp_best_worst.shape)
    # define X and y text
    X = yelp_best_worst["text"]
    y = yelp_best_worst["stars"]
    # define train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, test_size=0.25
    )
    # examine the object shapes
    print("X_train shape")
    print(X_train.shape)
    print("X_test shape")
    print(X_test.shape)
    # Part 2. Tokenize the text
    # use CountVextorizer() to create DTMs from X_train, X_test
    vect = CountVectorizer()
    # fit and transform X_train
    X_train_dtm = vect.fit_transform(raw_documents=X_train)
    # transform X_test
    X_test_dtm = vect.transform(raw_documents=X_test)
    # examine the object shapes
    # the rows are documents, the columns are terms
    # terms = tokens = features
    print("X_train_dtm shape")
    print(X_train_dtm.shape)
    print("X_test_dtm shape")
    print(X_test_dtm.shape)
    # examine the last 50 features
    print("Last 50 features")
    print(vect.get_feature_names_out()[-50:])
    # change various CountVectorizer() parameters
    # do not convert to lowercase
    vect = CountVectorizer(lowercase=False)
    X_train_dtm = vect.fit_transform(raw_documents=X_train)
    X_test_dtm = vect.transform(raw_documents=X_test)
    print("X_train_dtm shape for lowercase=False")
    print(X_train_dtm.shape)
    # change ngram_range: tuple(min_n, max_n)
    # the lower and upper boundary of the range of n values for different
    # n-grams to be extracted
    # all values of n such that min <= n <= max will be used
    # include 1-grams and 2-grams
    vect = CountVectorizer(ngram_range=(1, 2))
    X_train_dtm = vect.fit_transform(raw_documents=X_train)
    print("X_train_dtm shape for 1-grams, 2-grams")
    print(X_train_dtm.shape)
    # examine the last 50 features
    # any given 1-gram or 2-gram is used at least once, but we do not know
    # how many times they were used
    # it is not a matter of how many times the pairs (2-grams) occur,
    # but which terms occur next to one another in the actual documents
    print("Last 50 features")
    print(vect.get_feature_names_out()[-50:])
    # Part 3. Comparing the accuracy of different approaches
    # approach 1: the null model always predict the most frequent class
    # scikit-learn has "dummy classifier" to predict the most frequent class
    # the null model does no thinking, just predicts the most frequent class
    print("Calculate the null accuracy")
    print(y_test.value_counts().head(1) / y_test.shape)
    print(y_test.value_counts())
    # approach 2: use the default parameters of CountVectorizer
    # use the default parameters for CountVectorizer
    print("use the default parameters for CountVectorizer")
    vect = CountVectorizer()
    tokenize_test(
        vect=vect,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    # do not convert to lowercase
    print("do not convert to lowercase")
    vect = CountVectorizer(lowercase=False)
    tokenize_test(
        vect=vect,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    # include 1-grams and 2-grams
    print("include 1-grams and 2-grams")
    vect = CountVectorizer(ngram_range=(1, 2))
    tokenize_test(
        vect=vect,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    # Part 4. Removing frequent terms (stop words)
    # What: remove common words that appear in most documents
    # Why: they probably do not tell you much about your text
    # remove "english" stop words; known problems with "english"
    vect = CountVectorizer(stop_words="english")
    tokenize_test(
        vect=vect,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    # examine the stop words
    print("examine the stop words in 'english'")
    print(sorted(vect.get_stop_words()))
    # max_df is the maximum document frequency for a term to be allowed in
    # ignore terms that appear in more than 50 % of the documents
    # the lower the value, the more terms get excluded
    vect = CountVectorizer(max_df=0.5)
    tokenize_test(
        vect=vect,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    print("examine the stop words removed due to max_df (corpus specific)")
    print(sorted(vect.stop_words_))
    # Part 5. Removing infrequent terms
    # max_features: If not None, build a vocabulary that only considers
    # the top max_features ordered by term frequency across the corpus.
    # only keep the top 1000 most frequent terms
    # the intuition is that infrequent terms are useless
    print("only keep the top 1000 most frequent terms")
    vect = CountVectorizer(max_features=1000)
    tokenize_test(
        vect=vect,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    # min_df
    # When building the vocabulary, ignore terms that have a document
    # frequency strictly lower than the given threshold. This value is called
    # "cut-off" in the literature. If float, the parameter represents a
    # portion of documents. If integer, the parameter represents an
    # absolute count.
    # only keep terms that appear in at least two documents
    print("only keep terms that appear in at least two documents")
    vect = CountVectorizer(min_df=2)
    tokenize_test(
        vect=vect,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    # include 1-grams & 2-grams; keep terms that appear in at least 2 documents
    print(
        "include 1-grams & 2-grams; keep terms that appear in at least 2 documents"
    )
    vect = CountVectorizer(ngram_range=(1, 2), min_df=2)
    tokenize_test(
        vect=vect,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    # guidelines for tuning CountVectorizer
    # Use your knowledge of the problem, the text, and your understanding of
    # the tuning parameters to help you decide what parameters to tune and how
    # to tune them.
    # Experiment and let the data tell you the best approach.
    # Part 6. Handling Unicode errors
    # Text is made of characters. Files are made of bytes. Bytes represent
    # characters according to a particular encoding standard. To work with
    # text files in Python, their bytes must be decoded to a character set
    # called Unicode. Unicode is not an encoding. Unicode is a system that
    # assigns a unique number for every character in every language. These
    # numbers are called "code points." The code point for "A" is U+0041 and
    # the official name is "LATIN CAPITAL LETTER A." Unicode is a system of
    # code points. Encoding specifies how to store code points in memory.
    # The default encoding in Python 2 is ASCII. The default encoding in
    # Python 3 is UTF-8.
    print("Examine two types of strings")
    print('type(b"hello") where "b" is bytes (Unicode):', type(b"hello"))
    print('type("hello"):                             :', type("hello"))
    print("'decode' converts 'bytes' to 'str'")
    print(
        'b"hello".decode(encoding="utf-8")          :',
        b"hello".decode(encoding="utf-8"),
    )
    print("'encode' converts 'str' to 'bytes'")
    print(
        '"hello".encode(encoding="utf-8")           :',
        "hello".encode(encoding="utf-8"),
    )
    # read in a single Yelp review and use "latin-1" to read the file
    yelp_path_single = Path(
        "machine_learning_with_text_in_python/ML-text-main/data/yelp_single.csv"
    )
    yelp_single = ds.read_file(file_name=yelp_path, encoding="latin-1")


if __name__ == "__main__":
    main()
