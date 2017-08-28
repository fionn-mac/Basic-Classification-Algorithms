from sys import argv
import numpy as np
from collections import defaultdict

def read_train():
    with open(train_path) as f:
    # with open("datasets/q3/train.csv") as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            temp = line.split(",")
            if i > 0:
                for j, val in enumerate(temp):
                    if j < 6:
                        values[j].append(float(val))
                    elif j == 6:
                        values[j].append(int(val))
                    else:
                        values[j].append(val)
                indices.append(i-1)

            else:
                for j, val in enumerate(temp):
                    values.append([])
                    attributes[val] = j
                    if j < 6:
                        continuous.append(val)
                    elif j > 6:
                        discrete.append(val)
    return

def read_test():
    with open(test_path) as f:
    # with open("datasets/q3/test.csv") as f:
        correct = 0
        total = 0
        for i, line in enumerate(f):
            line = line.rstrip()
            temp = line.split(",")
            inAtt = []
            if i > 0:
                j = 0
                for val in temp:
                    if j == 6:
                        inAtt.append("Dummy Label")
                    if j < 6:
                        inAtt.append(float(val))
                    else:
                        inAtt.append(val)
                    j += 1
                classLabel = classify_test(inAtt, 0)
                print classLabel
                # total += 1
                # print classLabel, " ", label
                # if classLabel == label:
                #     correct += 1
    
    # print "Accuracy:",float(correct)*100/float(total-1),"%"
    return

def classify_test(attribute, node):
    # print node, " ", tree[node]
    if tree[node][0][0] == "continue":
        if tree[node][0][1] in discrete:
            for elem in tree[node]:
                # print attribute[attributes[elem[1]]], " ", elem[2]
                if attribute[attributes[elem[1]]] == elem[2]:
                    return classify_test(attribute, elem[3])
            
            count = [0, 0]
            for elem in tree[node]:
                # print elem
                retval = classify_test(attribute, elem[3])
                count[retval] += 1
            return 0 if count[0] > count[1] else 1
        else:
            elem = tree[node][0]
            # 
            if attribute[attributes[elem[1]]] < elem[2]:
                return classify_test(attribute, elem[3])
            else:
                return classify_test(attribute, elem[4])

    else:
        # print node, " ", tree[node]
        if tree[node][0][1] in discrete:
            for elem in tree[node]:
                # print attribute[attributes[elem[1]]], " ", elem[2]
                if attribute[attributes[elem[1]]] == elem[2]:
                    return elem[3]
            
            count = [0, 0]
            for elem in tree[node]:
                count[elem[3]] += 1
            return 0 if count[0] > count[1] else 1
        
        else:
            elem = tree[node][0]
            # 
            if attribute[attributes[elem[1]]] < elem[2]:
                return elem[3]
            else:
                return 1 - elem[3]

def generate_tree(indices):
    global node
    curr = node
    end = False
    quality = float(1)

    # Decide which attribute to split on
    for i in continuous:
        retval = entropy_continuous(values[attributes[i]], indices)
        if retval[1] <= quality:
            quality = retval[1]
            split_point = retval[0]
            attr = i

    for i in discrete:
        retval = entropy_discrete(values[attributes[i]], indices)
        if retval <= quality:
            quality = retval
            split_point = "discrete"
            attr = i

    if quality == float(0):
        end = True

    if split_point == "discrete":
        branches = []
        # Determine what values of the discrete attribute made it to this subtree
        for i in indices:
            if values[attributes[attr]][i] not in branches:
                branches.append(values[attributes[attr]][i])
                if end:
                    tree[curr].append(("end", attr, values[attributes[attr]][i], values[attributes["left"]][i]))

        for i, branch in enumerate(branches):
            direction = [] # Contains indices for each branch
            for index in indices:
                if values[attributes[attr]][index] == branch:
                    direction.append(index)

            if not end:
                node += 1
                tree[curr].append(("continue", attr, branch, node))
                generate_tree(direction)

    else:
        # Index matrices for each direction
        left = []
        right = []
        for i in indices:
            # 
            if values[attributes[attr]][i] < split_point:
                left.append(i)
            else:
                right.append(i)

        if end:
            tree[curr].append(("end", attr, split_point, values[attributes["left"]][left[0]]))
        else:
            node += 1
            leftDir = node
            generate_tree(left)
            node += 1
            rightDir = node
            generate_tree(right)
            tree[curr].append(("continue", attr, split_point, leftDir, rightDir))

    # print curr, " ", tree[curr]
    return

def entropy_discrete(data, indices):
    count = {}

    if len(indices) == 0:
        return 0
    
    for i in indices:
        if data[i] not in count:
            count[data[i]] = [0, 0]
        count[data[i]][values[attributes["left"]][i]] += 1
    
    quality = float(0)
    
    for i in count:
        p = float(count[i][0])/float(count[i][0] + count[i][1])
        
        if p > 0 and p < 1:
            quality += (-p*np.log2(p) - (1-p)*np.log2(1-p))*float(count[i][0] + count[i][1])/float(len(indices))

    return quality

def entropy_continuous(data, indices):
    minval = float(1)
    low = float(100000)
    high = float(0)
    
    for i in indices:
        if data[i] < low:
            low = data[i]
        if data[i] > high:
            high = data[i]

    retval = (high - low)/2
    step = float(high - low)/float(100)
    
    # if node == 0:
    #     print "Low", low, "High", high, "Step", step

    while low < high:
        left = [0, 0]
        right = [0, 0]
        entropy = [0, 0]

        for i in indices:
            #
            if data[i] < low:
                left[values[attributes["left"]][i]] += 1
            else:
                right[values[attributes["left"]][i]] += 1

        # if node == 0:
        #     print "Low", low, "Left", left, "Right", right

        left_count = float(left[0] + left[1])
        right_count = float(right[0] + right[1])

        if left_count != 0:
            p_left = float(left[0])/float(left[0] + left[1])
        
        if right_count != 0:
            p_right = float(right[0])/float(right[0] + right[1])
        
        if left[0] == 0 or left[0] == left_count:
            entropy[0] = 0
        else:
            entropy[0] = -p_left*np.log2(p_left) - (1 - p_left)*np.log2(1 - p_left)

        if right[0] == 0 or right[0] == right_count:
            entropy[1] = 0
        else:
            entropy[1] = -p_right*np.log2(p_right) - (1 - p_right)*np.log2(1 - p_right)

        quality = (entropy[0]*left_count + entropy[1]*right_count)/(left_count + right_count)

        if quality < minval:
            minval = quality
            retval = low

        low += step

    return retval, minval

def main():
    read_train()
    generate_tree(indices)

    read_test()

node = 0

values = []
indices = []
continuous = []
discrete = []

attributes = {}
tree = defaultdict(list)

train_path = argv[1]
test_path = argv[2]

main()