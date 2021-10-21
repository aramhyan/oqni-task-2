from sklearn.decomposition import PCA
from feature_creator import *
import pandas as pd

def get_diff(data, periods=1):
    data_diff = data.diff(periods=periods).fillna(0)
    data_diff.columns = [f'{i}_diff_{periods}' for i in data.columns]
    return data_diff


def add_features_data(data, data_feature, ws=5):
    for new_feature in ['iemg', 'ssi']:
        new_feature_name = f'{data_feature}_{new_feature}_{ws}'
        new_feature_array = get_feature(
            data[data_feature], ws, new_feature
        )
        data[new_feature_name] = new_feature_array
    return data

# train test splitting function
def split_data(X, y, train_size):
    # data splitting index
    split_index = split_index = int(train_size*len(X))
    x_train, y_train = X[:split_index], y[:split_index]
    x_test, y_test = X[split_index:], y[split_index:]
    return x_train, y_train, x_test, y_test 


def feature_engineering(data):
    data_new = pd.concat([data, get_diff(data), get_diff(data, periods=2)], axis=1)
    for column_name in data_new.columns:
        data_new = add_features_data(data_new, column_name)
#     data_new = data_new.drop(columns=['timestamp_milllisecs'])
    return data_new

def dim_reduction(x_train,x_test):
    clf = PCA(n_components=75)
    clf.fit(x_train)
    x_train = clf.transform(x_train)
    x_test = clf.transform(x_test)
    print('columns left:', x_train.shape[1])
    return x_train, x_test

# get sample weights
def get_sample_weights(y):
    counts = np.unique(y, return_counts=True)
    class_weights = [int(max(counts[1])/i) for i in counts[1]]
    weight_dict = dict(zip(counts[0], class_weights))
    sample_weight = [weight_dict[i] for i in y]
    return sample_weight

