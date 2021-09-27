import pandas as pd
import numpy as np
import joblib
import argparse

def load_and_process(df_name):
    data = pd.read_csv(df_name)
    data = data.fillna(0)
    data = data.drop(columns='timestamp_milllisecs')
    return data

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str)
    parser.add_argument('-model', type=str, default='xgb_experimental.joblib')
    args = parser.parse_args()
    arguments = vars(args)
    return arguments

def main():
    arguments = parser()
    testing_data = load_and_process(arguments['f'])
    model = joblib.load(arguments['model'])
    predictions = model.predict(testing_data)

    # print(predictions)
    # print(np.unique(predictions, return_counts=True))
    return predictions

if __name__=='__main__':
    main()

