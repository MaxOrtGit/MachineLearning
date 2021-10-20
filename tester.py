import numpy as np
import NeuralNetworkOld as nno
y = []

#39-10
for x in range(39, 9, -1):
    y.append(x)

print(y)
print(len(y))

training_images = y[:int(len(y) / 2)]
print(training_images)
print(len(training_images))
testing_images = y[int(len(y) / 2):]
print(testing_images)
print(len(testing_images))

predictions = [[2, 3, 4, 5], [2, 2, 2, 2], [1, 1, 1, 1], [6, 6, 6, 6]]
labels = [[5, 3, 2, 4], [2, 2, 2, 2], [2, 2, 2, 2], [3, 3, 9, 9]]

# array3 = [((a - b) ** 2) for a, b in zip(predictions, labels)]
array3 = np.subtract(predictions, labels)
array4 = np.square(np.subtract(predictions, labels))

print(array3)
print(array4)
print([np.average(k) for k in array4])

ary1 = [[2, 2], [1, 1]]
ary2 = [4, 2, 1, 3]

print(np.add(np.concatenate(ary1), ary2))
print()
print()
print()


def unison_shuffled_copies(aye, buh):
    assert len(aye) == len(buh)
    p = np.random.permutation(len(aye))
    return np.array(aye)[p], np.array(buh)[p]


print(unison_shuffled_copies(predictions, labels))

for batchNumber in range(0, int(np.ceil(len(predictions) / 1))):
    print(batchNumber)

layer_sizes = (4, 3, 2, 1)

net = nno.NeuralNetwork(layer_sizes)

tesdt = (1, 2)
tesdt = np.add(tesdt, (4, 5))
print(tesdt)

print(len(range(4,10)))

for x in range(4,10):
    print(x)
    

sum = [np.sum(sum, weight_val) for weight_val in y]
print(sum)