from sys import argv
import numpy as np
from copy import deepcopy

def read_train(train_path, data, label):
    with open(train_path) as f:
        i = 0
        for line in f:
            if line == "\n":
                break
            temp = line.split(",")

            if '?' not in temp:
                temp = map(int, temp)
                label.extend(temp[-1:])
                label[i] -= 3
                attributes = [1]
                attributes.extend(temp[1:-1])
                data.append(label[i]*np.array(attributes))
                i += 1
    return

def calc_norm(data, norm):
    for i, sample in enumerate(data):
        norm[i] = np.dot(sample, sample)
    return

def pass_over(data, norm, W, margin, n):
    count = 0
    for i, sample in enumerate(data):
        value = np.dot(W, sample) - margin
        if value < 0:
            value /= norm[i]
            count += 1
            np.add(W, -n*value*sample, out=W)

    accuracy = float(len(data) - count)/float(len(data))
    print "Training Accuracy: " + str(accuracy*100) + "%"

    return

def train(data, norm, W, margin):
    n = 1e-1
    
    for epoch in xrange(5000):
        pass_over(data, norm, W, margin, n)

    return

def validate(data, W, margin):
    count = 0

    for sample in data:
        if np.dot(W, sample) <= 0:
            count += 1

    accuracy = float(len(data) - count)*100/float(len(data))
    print "Validation Accuracy: " + str(accuracy) + "%"

    return accuracy

def test(data, W, margin):
    count = 0
    for sample in data:
        predictLabel = 2
        if np.dot(W, sample) > 0:
            predictLabel = 4

        print predictLabel
    return

if __name__ == '__main__':

    data = []
    label = []
    data_test = []
    np.random.seed()

    if len(argv) != 2:
        print "Usage: python q1.py [relative/path/to/file]"

    # Read training and testing data from file
    read_train(argv[1], data, label)

    # Separate into Training and Validation sets
    i = int(len(data)*0.8)
    data_train = data[:i]
    data_validate = data[i+1:]
    norm_train = np.zeros(len(data_train))

    calc_norm(data_train, norm_train)

    # Single Perceptron with Relaxation and Margin; May not converge
    margin = 1e-6
    W = np.array(np.random.uniform(-1, 1, len(data[0]))) #Initialize Weight Matrix
    # W = np.zeros(len(data[0]))
    train(data_train, norm_train, W, margin) #Train model
    accuracy = validate(data_validate, W, margin) #Validate model

