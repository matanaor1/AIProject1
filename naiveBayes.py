import abstractClassifier as super
from sklearn.naive_bayes import ComplementNB


class NBClassifier(super.abstract_classifier):

    def __init__(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels
        self.nb_Member = ComplementNB()

    def train(self): # after this function the ComplementNB is ready to classify
        self.nb_Member.fit(self.train_features, self.train_labels)

    def classify(self, newVector):
        return self.nb_Member.predict(newVector)
