from sys import argv
import numpy as np
from copy import deepcopy

def read_train(data, label):
    with open(train_path) as f:
        for i, line in enumerate(f):
            factor = 1
            temp = map(int, line.split(","))
            label.append(temp[0])
            attributes = [1]
            attributes.extend(temp[1:])
            if temp[0] == 0:
                factor = -1;
            data.append(factor*np.array(attributes))
    return

def read_test(data):
    with open(test_path) as f:
        for i, line in enumerate(f):
            temp = map(int, line.split(","))
            attributes = [1]
            attributes.extend(temp)
            data.append(np.array(attributes))
    return

def single_perceptron(data, label, W, margin):
    count = 0
    for sample in data:
        if np.dot(W, sample)/np.dot(sample, sample) <= margin:
            np.add(W, sample, out=W)
            count += 1

    return

def batch_perceptron(data, label, W, margin):
    count = 0
    loss = np.zeros(len(data[0]))

    for i, sample in enumerate(data):
        if np.dot(W, sample)/np.dot(sample, sample) <= margin:
            np.add(loss, sample, out=loss)
            count += 1

    np.add(loss, W, out=W)
    return

def train(data, label, W, margin, single):
    flag = True
    
    for epoch in xrange(1000):
        if single:
            single_perceptron(data, label, W, margin)
        else:
            batch_perceptron(data, label, W, margin)

    return

def validate(data, label, W, margin):
    count = 0
    
    for sample in data:
        if np.dot(W, sample) <= 0:
            count += 1

    return float(len(data) - count)*100/float(len(data))

def test(data, W, margin):
    for sample in data:
        predictLabel = 0
        if np.dot(W, sample) > 0:
            predictLabel = 1

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
    label_train = label[:i]
    data_validate = data[i+1:]
    label_validate = label[i+1:]
    single = 1

    for i in xrange(4):
        maxAccuracy = float(0)
        marginFinal = 0
        if i % 2:
            # Single and Batch with Margin; may not converge
            for margin in np.arange(0.001, 0.101, 0.05):
                W = np.zeros(len(data[0]))
                train(data_train, label_train, W, margin, single) #Train model
                accuracy = validate(data_validate, label_validate, W, margin) #Validate model
                if accuracy >= maxAccuracy:
                    WFinal = deepcopy(W)
                    marginFinal = margin
                    maxAccuracy = accuracy

            # print marginFinal, maxAccuracy
            test(data_test, WFinal, marginFinal)

        else:
            WFinal = np.zeros(len(data[0]))
            # Single and Batch without Margin
            train(data, label, WFinal, marginFinal, single) #Train model
            # maxAccuracy = validate(data_validate, label_validate, WFinal, marginFinal) #Validate model

            test(data_test, WFinal, marginFinal)

        if i == 1:
            single = 0

    return

train_path = argv[1]
test_path = argv[2]

main()