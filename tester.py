import numpy as np

y = []
for x in range(30, -1, -1):
    y.append(x)

print(y)

training_images = y[:int(len(y) / 2)]
print(training_images)
testing_images = y[int(len(y) / 2):]
print(testing_images)

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

for batchNumber in range(0, int(np.ceil(len(predictions) / 1))):
    print(batchNumber)