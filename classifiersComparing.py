import csv

import pandas as pd
import randomForest as RF
import Perceptron as PE
import naiveBayes as NB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    train_featurs = pd.read_table('train_x_after_fs.csv', sep=',', header=0)
    train_labels = pd.read_csv('train_y.csv', sep='\n', header=0)
    validation_features =  pd.read_csv('val_x_after_fs.csv', sep=',', header=0)
    validation_labels = pd.read_csv('val_y.csv', sep=',', header=0)
    f = lambda x: 1 if x >= 3.5 else 0
    train_labels = train_labels.applymap(f)
    validation_labels = validation_labels.applymap(f)


    rf_cls = RF.RFClassifier(train_featurs, train_labels, 100)
    rf_cls.train()
    perceptron_cls = PE.perceptronClassifier(train_featurs, train_labels)
    perceptron_cls.train()
    nb_cls = NB.NBClassifier(train_featurs, train_labels)
    nb_cls.train()

    rf_results = rf_cls.classify(validation_features)
    percaptron_results = perceptron_cls.classify(validation_features)
    nb_results = nb_cls.classify(validation_features)

    rf_hit_rate = accuracy_score(validation_labels, rf_results)
    perceptron_hit_rate = accuracy_score(validation_labels, percaptron_results)
    nb_hit_rate = accuracy_score(validation_labels, nb_results)
    print(rf_hit_rate)
    print(perceptron_hit_rate)
    print(nb_hit_rate)

    #draw the first graph:
    objects = ('random forest', 'perceptron', 'naive bayes')
    y_pos = np.arange(len(objects))
    performance = [rf_hit_rate, perceptron_hit_rate, nb_hit_rate]
    plt.bar(y_pos, performance, align='center')
    plt.xticks(y_pos, objects)
    plt.ylabel('hit rate')
    plt.title('comparing classifiers hit rate')
    plt.show()

    