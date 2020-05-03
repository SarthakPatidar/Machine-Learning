import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

dataset = {'k': [[1,2], [2,3], [3,1]], 'r': [[6,5], [7,7], [8,9]]}
new_features = [5,7]

def k_nearest_neighbor(dataset, predict, k=3):
    distances = []
    for group in dataset:
        for features in dataset[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    neighbors = [i[1] for i in sorted(distances)[:k]]
    return Counter(neighbors).most_common(1)[0][0]

k_nearest_neighbor(dataset, new_features, k=3)
predicted_class = k_nearest_neighbor(dataset, new_features, k=3)

[[plt.scatter(j[0], j[1], color=i) for j in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], color=predicted_class)
plt.show()
    
    


