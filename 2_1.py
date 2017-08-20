import numpy as np

def read(data, label):
    with open("datasets/q2_breast_cancer.train") as f:
        i = 0
        for line in f:
            if line == "\n":
                break
            temp = line.split(",")
            if '?' not in temp:
                temp = map(int, temp)
                label.extend(temp[-1:])
                label[i] -= 3
                data.append(np.array(temp[1:-1]))
                i += 1
    return
    
def pass_over(data, label, W, b, n):
    count = 0
    # loss = np.zeros(len(data[0]))
    for i, sample in enumerate(data):
        g_x = np.sign(np.dot(W, sample))
        if g_x != label[i]:
            coeff = n*(g_x - label[i])/np.inner(sample, sample)
            np.subtract(W, coeff*sample, out=W)
            count += 1

    accuracy = float(len(data) - count)/float(len(data))
    # print "W", W
    # np.add(W, loss, out=W)
    print "Training Accuracy :", accuracy
    return

def train(data, W, label):
    epoch = 1
    b = 0
    n = 0.001
    
    while epoch < 1500:
        print "epoch :", epoch
        pass_over(data, label, W, b, n)
        epoch += 1
    
    return

def validate(data, W, label):
    count = 0
    
    for i, sample in enumerate(data):
        if np.sign(np.dot(W, sample)) != np.sign(label[i]):
            count += 1
    
    print "Validation Accuracy :", float(len(data) - count)/float(len(data))
    
    return

def main():
    data = []
    label = []
    np.random.seed()

    # Read training data from file
    read(data, label)
    W = np.array(np.random.uniform(-1, 1, len(data[0])))
    # Separate into Training and Validation sets
    i = int(len(data)*0.8)
    data_train = data[:i]
    label_train = label[:i]
    
    data_validate = data[i+1:]
    label_validate = label[i+1:]
    
    train(data_train, W, label) #Train model
    validate(data_validate, W, label) #Validate model
    
    return

main()