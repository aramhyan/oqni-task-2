import pandas as pd
import numpy as np
import joblib
import argparse
from feature_creator import *

def get_diff(data, periods=1):
    data_diff = data.diff(periods=periods).fillna(0)
    data_diff.columns = [f'{i}_diff_{periods}' for i in data.columns]
    return data_diff

def add_features_data(data, data_feature, ws=5):
    for new_feature in ['iemg', 'mav', 'mav1', 'mav2', 'ssi']:
        new_feature_name = f'{data_feature}_{new_feature}_{ws}'
        new_feature_array = get_feature(
            data[data_feature], ws, new_feature
        )
        data[new_feature_name] = new_feature_array
    return data


def feature_engineering(data):
    data_new = pd.concat([data, get_diff(data), get_diff(data, periods=2)], axis=1)
    data_new = add_features_data(data_new, 'ch6_diff_1')
    return data_new


def load_and_process(df_name):
    data = pd.read_csv(df_name)
    data = data.fillna(0)
    data = data.drop(columns='timestamp_milllisecs')
    data = feature_engineering(data)
    return data

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str)
    parser.add_argument('-model', type=str, default='../models/xgb_weighted_samples.joblib')
    args = parser.parse_args()
    arguments = vars(args)
    return arguments

def main():
    arguments = parser()
    testing_data = load_and_process(arguments['f'])
    model = joblib.load(arguments['model'])
    predictions = model.predict(testing_data)

    np.save('xgb_weighted_samples_predictions.npy', predictions)
    return predictions

if __name__=='__main__':
    main()

