import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.features import RFECV



if __name__ == '__main__':
    data_x = pd.read_csv('train_x.csv', sep=',', skipinitialspace=True, header=0)

    data_y = pd.read_csv('train_y.csv', sep=',', skipinitialspace=True, header=0)
    bestfeatures = SelectKBest(k=28)
    fit = bestfeatures.fit(data_x,data_y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(data_x.columns)
    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    #print(featureScores.nlargest(28,'Score'))  #print best features
    # Create and fit selector
    selector = SelectKBest(k=28)
    selector.fit(data_x, data_y)
    # Get columns to keep
    cols = selector.get_support()
    # Create new dataframe with only desired columns, or overwrite existing
    new_features = data_x.columns[cols]
    data_x=data_x[new_features]
    data_x.to_csv("train_x_after_fs.csv", sep=',', header=True, index=False)

    val_x = pd.read_csv('val_x.csv', sep=',', skipinitialspace=True, header=0)
    val_x = val_x[new_features]
    val_x.to_csv("val_x_after_fs.csv", sep=',', header=True, index=False)

    test_x = pd.read_csv('test_x.csv', sep=',', skipinitialspace=True, header=0)
    test_x = test_x[new_features]
    test_x.to_csv("test_x_after_fs.csv", sep=',', header=True, index=False)


"""
    #RFECV test for optimal # of features
    f= lambda x: int(2*x)
    data_y=data_y.applymap(f)
    viz = RFECV(SVC(kernel='linear', C=1))
    viz.fit(data_x, data_y)
    viz.poof()
"""