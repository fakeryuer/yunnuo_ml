# -- coding utf-8 --
# Automatically generated simple classification code for algorithm mlp2
from __future__ import print_function
import softcomai as sai
from threading import Thread
from time import sleep
from sklearn.metrics import 
from sklearn.neural_network import MLPClassifier
import moxing as mox
import os
import pickle
import pandas as pd
import numpy as np


# Get parameters from training configuration
def get_params()
    train_data=sai.context.param(train_data)
    validation_data=sai.context.param(validation_data)
    label_column=train_data['label']
    epoch = int(sai.context.param('epoch'))
    return train_data, validation_data, label_column, epoch


# Load data by dataset entity name
def load_data(dataInfo, label_column)
    data = sai.data_reference.get_data_reference(dataInfo['dataset'], dataInfo['entity']).to_pandas_dataframe()
    y = data[label_column]
    X = data.drop([label_column], axis=1)
    return X, y


# Train model by data
def train_model(x, y, epoch, logs)
    clf = MLPClassifier(activation='relu', alpha=1e-05, solver='adam',batch_size=500, power_t=0.5,
                        epsilon=1e-08, hidden_layer_sizes=(200, 200), random_state=1, beta_1=0.9,
                        beta_2=0.999, learning_rate='constant', momentum=0.9,  learning_rate_init=0.001,
                        nesterovs_momentum=True, shuffle=True, tol=0.0001, max_iter=1,
                        validation_fraction=0.1, verbose=False, warm_start=True)
    if not epoch or epoch = 0
        epoch = 20
    epochNum = 0
    while int(epochNum)  int(epoch)
        clf.fit(x, y)
        pred_y = clf.predict(x)
        logs.log_metric(loss, clf.loss_)
        logs.log_metric(accuracy, accuracy_score(y, pred_y))
        logs.log_metric(f1, f1_score(y, pred_y, average='macro'))
        logs.log_metric(precision,precision_score(y, pred_y, average='macro'))
        logs.log_metric(recall,recall_score(y, pred_y, average='macro'))
        print(clf.loss_)
        print('epochNum ', epochNum)
        epochNum = epochNum + 1

    return clf

# Score the model
def score_model(y_true, y_pred, logs)
    accuracy = accuracy_score(y_true, y_pred)
    f1_avg = f1_score(y_true, y_pred, average='macro')
    precision_avg = precision_score(y_true, y_pred, average='macro')
    recall_avg = recall_score(y_true, y_pred, average='macro')
    logs.log_property(f1_avg, f1_avg)
    logs.log_property(precision_avg,precision_avg)
    logs.log_property(recall_avg,recall_avg)


# Save the model
def save_model(clf)
    with open(os.path.join(sai.context.param(sai.context.MODEL_PATH), 'model_clf.pkl'), 'wb') as mf
        pickle.dump(clf, mf)


# Run
def main()
    with sai.report(True) as logs
        train_data, validation_data, label_column, epoch = get_params()
        train_X, train_Y = load_data(train_data, label_column)
        validation_X, validation_Y = load_data(validation_data, label_column)
        clf = train_model(train_X, train_Y, epoch, logs)
        score_model(validation_Y, clf.predict(validation_X), logs)
        save_model(clf)

if __name__ == '__main__'
    main()