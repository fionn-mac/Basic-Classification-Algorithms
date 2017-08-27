import numpy as np
from copy import deepcopy

def read(data, label):
    with open("datasets/q2/train.csv") as f:
        i = 0
        for line in f:
            if line == "\n":
                break
            temp = line.split(",")
            if '?' not in temp:
                temp = map(int, temp)
                label.extend(temp[-1:])
                label[i] -= 3
                attributes = temp[1:-1]
                data.append(label[i]*np.array(attributes))
                i += 1
    return
    
def pass_over(data, W, margin, n):
    count = 0
    loss = np.zeros(len(data[0]))
    for sample in data:
        value = float(np.dot(W, sample) - margin)/float(np.linalg.norm(sample))
        if value <= 0:
            np.add(loss, n*value*sample, out=loss)
            count += 1

    accuracy = float(len(data) - count)/float(len(data))
    np.subtract(W, loss, out=W)
    print "Training Accuracy :", accuracy
    return

def train(data, W, margin):
    n = 0.001
    
    for epoch in xrange(1500):
        pass_over(data, W, margin, n)

    return

def validate(data, W, margin):
    count = 0
    
    for i, sample in enumerate(data):
        if float(np.dot(W, sample) - margin)/float(np.linalg.norm(sample)) <= 0:
            count += 1

    accuracy = float(len(data) - count)*100/float(len(data))
    print "Validation Accuracy :", accuracy, "where margin :", margin

    return accuracy

def main():
    data = []
    label = []
    # data_test = []
    # label_test = []
    np.random.seed()

    # Read training data from file
    read(data, label)
    
    # Separate into Training and Validation sets
    i = int(len(data)*0.8)
    data_train = data[:i]
    data_validate = data[i+1:]
    single = 1
    
    for i in xrange(2):
        maxAccuracy = float(0)
        marginFinal = 0
        if i:
            pass
        else:
            # Single and Batch with Margin; may not converge
            for margin in np.arange(0, 10, 0.5):
                W = np.array(np.random.uniform(-1, 1, len(data[0])))
                train(data_train, W, margin) #Train model
                accuracy = validate(data_validate, W, margin) #Validate model
                if accuracy >= maxAccuracy:
                    WFinal = deepcopy(W)
                    marginFinal = margin
                    maxAccuracy = accuracy
            
            print "Accuracy :", maxAccuracy, "Margin: ", marginFinal
            # validate(data_test, label_test, WFinal, marginFinal)

    return

main()