from sys import argv
import numpy as np
from copy import deepcopy

def read_train(data, label):
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

def read_test(data):
    with open(test_path) as f:
        for line in f:
            if line == "\n":
                break
            temp = line.split(",")
            if '?' not in temp:
                temp = map(int, temp)
                attributes = [1]
                attributes.extend(temp[1:])
                data.append(np.array(attributes))
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

    return

def train(data, norm, W, margin):
    n = 1e-1
    
    for epoch in xrange(5000):
        pass_over(data, norm, W, margin, n)

    return

def validate(data, W, margin):
    count = 0

    for sample in data:
        if np.dot(W, sample) <= margin:
            count += 1

    accuracy = float(len(data) - count)*100/float(len(data))

    return accuracy

def test(data, W, margin):
    count = 0
    for sample in data:
        predictLabel = 2
        if np.dot(W, sample) > margin:
            predictLabel = 4

        print predictLabel
    return

def main():
    data = []
    label = []
    data_test = []

    np.random.seed()

    # Read training and testing data from file
    read_train(data, label)
    read_test(data_test)

    # Separate into Training and Validation sets
    i = int(len(data)*0.8)
    data_train = data[:i]
    data_validate = data[i+1:]
    norm_train = np.zeros(len(data_train))

    calc_norm(data_train, norm_train)

    single = 1
    
    for i in xrange(2):
        maxAccuracy = float(0)
        marginFinal = 0
        if i:
            pass
        else:
            # Single Perceptron with Relaxation and Margin; May not converge
            for margin in np.arange(0, 0.01, 0.001):
                W = np.array(np.random.uniform(-1, 1, len(data[0]))) #Initialize Weight Matrix

                train(data_train, norm_train, W, margin) #Train model
                accuracy = validate(data_validate, W, margin) #Validate model
                if accuracy >= maxAccuracy:
                    WFinal = deepcopy(W)
                    marginFinal = margin
                    maxAccuracy = accuracy

            test(data_test, WFinal, marginFinal)

    return

train_path = argv[1]
test_path = argv[2]

main()