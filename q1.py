import numpy as np
from copy import deepcopy

def read(data, label):
    with open("datasets/q1/train.csv") as f:
        for i, line in enumerate(f):
            factor = 1
            temp = map(int, line.split(","))
            label.append(temp[0])
            attributes = np.array(temp[1:])
            if temp[0] == 0:
                factor = -1;
            data.append(factor*attributes)
    return

def single_perceptron(data, label, W, margin):
    count = 0
    for sample in data:
        if float(np.dot(W, sample))/float(np.linalg.norm(sample)) - margin <= 0:
        # if np.dot(W, sample) - margin <= 0:
            np.add(W, sample, out=W)
            count += 1
    
    # print "Training Accuracy :", float(len(data) - count)*100/float(len(data)), "%"
    return

def batch_perceptron(data, label, W, margin):
    count = 0
    loss = np.zeros(len(data[0]))
    
    for i, sample in enumerate(data):
        if float(np.dot(W, sample))/float(np.linalg.norm(sample)) - margin <= 0:
        # if np.dot(W, sample) - margin <= 0:
            np.add(loss, sample, out=loss)
            count += 1
    
    # print "Training Accuracy :", float(len(data) - count)/float(len(data))
    np.add(loss, W, out=W)
    return

def train(data, label, W, margin, single):
    flag = True
    
    for epoch in xrange(20):
        if single:
            single_perceptron(data, label, W, margin)
        else:
            batch_perceptron(data, label, W, margin)

    return

def validate(data, label, W, margin):
    count = 0
    
    for sample in data:
        if float(np.dot(W, sample))/float(np.linalg.norm(sample)) - margin <= 0:
            count += 1

    # print "Validation Accuracy :", float(len(data) - count)/float(len(data))
    return float(len(data) - count)*100/float(len(data))

def main():
    data = []
    label = []
    data_test = []
    label_test = []
    
    np.random.seed()

    # Read training data from file
    read(data, label)
    
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
            for margin in np.arange(200, 4001, 50):
                W = np.zeros(784)
                train(data_train, label_train, W, margin, single) #Train model
                accuracy = validate(data_validate, label_validate, W, margin) #Validate model
                if accuracy >= maxAccuracy:
                    WFinal = deepcopy(W)
                    marginFinal = margin
                    maxAccuracy = accuracy
            
            if single:
                print "Single -- Accuracy :", maxAccuracy, "Margin: ", marginFinal
            else:
                print "Batch -- Accuracy :", maxAccuracy, "Margin: ", marginFinal
            # validate(data_test, label_test, WFinal, marginFinal)

        else:
            WFinal = np.zeros(784)
            # Single and Batch without Margin
            train(data_train, label_train, WFinal, marginFinal, single) #Train model
            maxAccuracy = validate(data_validate, label_validate, WFinal, marginFinal) #Validate model

            if single:
                print "Single -- Accuracy :", maxAccuracy, "Margin: ", marginFinal
            else:
                print "Batch -- Accuracy :", maxAccuracy, "Margin: ", marginFinal
            # validate(data_test, label_test, WFinal, marginFinal)
        
        if i == 1:
            single = 0

    return

main()