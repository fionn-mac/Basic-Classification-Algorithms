import numpy as np

def read(data, label):
    with open("datasets/mnist_train.csv") as f:
        for i, line in enumerate(f):
            temp = map(int, line.split(","))
            label.append(temp[0])
            data.append(np.array(temp[1:]))
    return

def pass_over(data, label, W):
    flag = False
    count = 0
    loss = np.zeros(len(data[0]))
    
    for i, sample in enumerate(data):
        g_x = 1 if np.dot(W, sample) > 0 else 0
        if label[i] != g_x:
            if g_x == 0:
                np.add(loss, sample, out=loss)
            else:
                np.subtract(loss, sample, out=loss)
            count += 1
            flag = True
    
    print "Training Accuracy :", float(len(data) - count)/float(len(data))
    np.add(loss, W, out=W)
    return flag

def train(data, label, W):
    flag = True
    epoch = 1
    
    while flag:
        print "epoch :", epoch
        flag = pass_over(data, label, W)
        epoch += 1
    
    return

def validate(data, label, W):
    count = 0
    
    for i, sample in enumerate(data):
        g_x = 1 if np.dot(W, sample) > 0 else 0
        if label[i] != g_x:
            count += 1
    
    print "Validation Accuracy :", float(len(data) - count)/float(len(data))
    
    return

def main():
    data = []
    label = []
    np.random.seed()
    W = np.array(np.random.uniform(-1, 1, 784))
    
    # Read training data from file
    read(data, label)
    
    # Separate into Training and Validation sets
    i = int(len(data)*0.8)
    data_train = data[:i]
    label_train = label[:i]
    data_validate = data[i+1:]
    label_validate = label[i+1:]
    
    train(data_train, label_train, W) #Train model
    validate(data_validate, label_validate, W) #Validate model
    
    return

main()