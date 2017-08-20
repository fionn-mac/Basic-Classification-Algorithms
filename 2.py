import numpy as np

def read(data, label):
    with open("datasets/q2_breast_cancer.train") as f:
        for i, line in enumerate(f):
            temp = map(int, line.split(","))
            label.append(temp[0])
            data.append(np.array(temp[1:]))
    return