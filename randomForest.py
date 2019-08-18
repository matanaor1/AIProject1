from sklearn.ensemble.forest import RandomForestClassifier
import abstractClassifier as super

class RFClassifier(super.abstract_classifier):

    def __init__(self, train_features, train_labels, num_of_trees):
        self.train_features = train_features
        self.train_labels = train_labels
        self.rf_member = RandomForestClassifier(num_of_trees)

    def train(self):
        self.rf_member.fit(self.train_features, self.train_labels)

    def classify(self, newVector):
        return self.rf_member.predict(newVector)