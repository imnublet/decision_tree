import numpy as np
import pandas as pd


avocados = pd.DataFrame(data={
    'green':     [1, 1, 1, 1, 0, 0, 0, 0],
    'brown':     [1, 1, 0, 1, 1, 1, 0, 1],
    'firmness':  [1, 1, 0, 1, 1, 0, 1, 1],
    'softness':  [0, 1, 1, 1, 1, 1, 1, 1],
    'nub_loose': [0, 1, 1, 1, 0, 1, 1, 0],
    'ripe':      [0, 1, 0, 1, 0, 0, 1, 1]
})


def gini(labels):
    label_freq = {}
    temp = 0
    for i in labels:
        if i in label_freq:
            label_freq[i] += 1
        else:
            label_freq[i] = 1
    for label in label_freq.values():
        temp += (label/len(labels))**2
    return round(1-temp,3)

# print(avocados)
# label_portions = []
# label_portions.append(avocados.loc[avocados['firmness'] == 1].index.tolist())
for feature in avocados:
    yes_indeces = avocados[feature].loc[avocados[feature] == True]
    print(yes_indeces)
    print(len(yes_indeces))
    print(gini(yes_indeces))

# print(type(label_portions))
# print(label_portions)
# print(gini(label_portions[0]))