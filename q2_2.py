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

def modified_perceptron(data, W):
    accuracy = 0.0
    flag = False

    for epoch in xrange(1000):
        count = 0
        for i, sample in enumerate(data):
            if np.dot(W, sample) < 0:
                np.add(W, (1 - accuracy)*sample, out=W)
                count += 1
        accuracy = float(i - count + 1)/float(i+1)
        print "Training Accuracy: " + str(accuracy)
        if count == 0:
            flag = True
            break
    
    if flag:
        print "Converged at Epoch: " + str(epoch+1)
    else:
        print "Did not converge after Epoch: " + str(epoch+1)

    return

def validate(data, W):
    count = 0

    for sample in data:
        if np.dot(W, sample) <= 0:
            count += 1

    accuracy = float(len(data) - count)*100/float(len(data))
    print "Validation Accuracy: " + str(accuracy)

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

    W = np.array(np.random.uniform(-1, 1, len(data_train[0]))) #Initialize Weight Matrix
    modified_perceptron(data_train, W)
    validate(data_validate, W)
