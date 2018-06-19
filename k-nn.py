import sys
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import NearestNeighbors
from KNN_implementation import kd_tree

data = pd.read_csv("anime.csv")

# Dealing with unknown values
data.loc[(data['episodes'] == 'Unknown'), 'episodes'] = (data['episodes'] != "Unknown").median()
data['episodes'] = data['episodes'].astype(float)

data['rating'] = data['rating'].astype(float)

data['rating'].fillna(data['rating'].median(), inplace=True)

data['members'] = data['members'].astype(float)

features = pd.concat([data[['rating']], data[['type']], data[['members']], pd.get_dummies(data[["type"]]),
                      data["genre"].str.get_dummies(sep=",")], axis=1)

data['name'] = data['name'].map(lambda name: re.sub('[^A-Za-z0-9]+', " ", name))

max_abs_scaler = MaxAbsScaler()
features = max_abs_scaler.fit_transform(features)

nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(features)
distances, indices = nbrs.kneighbors(features)


def get_index_from_name(name):
    return data[data["name"] == name].index.tolist()[0]


print(data[data['id'] in indices[get_index_from_name(sys.argv[1])]].name)
