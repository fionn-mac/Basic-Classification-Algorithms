from sys import argv
import numpy as np
from copy import deepcopy

def read(file_path, data, label):
    with open(file_path) as f:
        for i, line in enumerate(f):
            factor = 1
            temp = map(int, line.split(","))
            label.append(temp[0])
            attributes = np.array([1] + temp[1:])
            if temp[0] == 0:
                factor = -1;
            data.append(factor*attributes)
    return

def perceptron(data, W, margin, batch_size):
    count = 0
    loss = np.zeros(len(data[0]))

    for i, sample in enumerate(data):
        if np.dot(W, sample)/np.dot(sample, sample) <= margin:
            np.add(loss, sample, out=loss)
            count += 1

        if (i+1)%batch_size == 0:
            np.add(loss, W, out=W)
            loss = np.zeros(len(data[0]))

    np.add(loss, W, out=W)
    return float(len(data) - count)/float(len(data))

def train(data, W, margin, batch_size):
    flag = True
    for epoch in xrange(1000):
            accuracy = perceptron(data, W, margin, batch_size)
            print "Training Accuracy: " + str(accuracy*100) + "%"
            if int(accuracy):
                flag = False
                print "Converged at Epoch: " + str(epoch+1) + " with Margin: " + str(margin) + " and Batch Size: " + str(batch_size)
                break

    if flag:
        print "Did not converge after Epoch: " + str(epoch+1) + " with Margin: " + str(margin) + " and Batch Size: " + str(batch_size)
    return

def validate(data, label, W, margin):
    count = 0
    for i, sample in enumerate(data):
        predictLabel = 1
        if np.dot(W, sample) <= 0:
            predictLabel = 0

        if predictLabel == label[i]:
            count += 1

    return float(count)*100/float(i+1)

if __name__ == '__main__':

    data = []
    label = []
    np.random.seed()

    if len(argv) != 2:
        print "Usage: python q1.py [relative/path/to/file]"

    # Read training data from file
    read(argv[1], data, label)
    
    # Separate into Training and Validation sets
    i = int(len(data)*0.8)
    data_train = data[:i]
    data_validate = data[i+1:]
    label_validate = label[i+1:]

    for i, sample in enumerate(data_validate):
        if label_validate[i] == 0:
            data_validate[i] = -1*data_validate[i]

    weight_length = len(data[0])

    print "Enter Batch Size (Must lie between 1 and " + str(len(data_train)) + ")"
    
    while(1):
        batch_size = input()
        if batch_size > 0 and batch_size <= len(data_train):
            break
        else:
            print "Please enter a valid Batch Size"

    maxAccuracy = float(0)

    # Perceptron Algorithm with Margin; May not converge
    for margin in np.arange(0.50, 1, 1):
        W = np.zeros(weight_length)
        train(data_train, W, margin, batch_size) #Train model
        accuracy = validate(data_validate, label_validate, W, margin) #Validate model
        if accuracy >= maxAccuracy:
            WFinal = deepcopy(W)
            marginFinal = margin
            maxAccuracy = accuracy

    if batch_size == 1:
        print "Single -- Accuracy :", maxAccuracy, "Margin: ", marginFinal, "\n"
    else:
        print "Batch -- Accuracy :", maxAccuracy, "Margin: ", marginFinal, "\n"

    # Perceptron algorithm without Margin
    marginFinal = 0
    WFinal = np.zeros(weight_length)
    train(data_train, WFinal, marginFinal, batch_size) #Train model
    maxAccuracy = validate(data_validate, label_validate, WFinal, marginFinal) #Validate model
    
    if batch_size == 1:
        print "Single -- Accuracy :", maxAccuracy, "Margin: ", marginFinal, "\n"
    else:
        print "Batch -- Accuracy :", maxAccuracy, "Margin: ", marginFinal, "\n"
