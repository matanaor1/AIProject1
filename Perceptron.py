import abstractClassifier as super
from sklearn.linear_model import Perceptron


class perceptronClassifier(super.abstract_classifier):

  def __init__(self, train_features, train_labels):
    self.train_features = train_features
    self.train_labels = train_labels
    self.perceptron_member = Perceptron()

  def train(self): # after this function the ComplementNB is ready to classify
      self.perceptron_member.fit(self.train_features, self.train_labels)

  def classify(self, newVector):
    return self.perceptron_member.predict(newVector)