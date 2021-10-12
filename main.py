import NeuralNetworkOld as nno
import NetworkR as nn
import numpy as np
import matplotlib.pyplot as plt

with np.load('mnist.npz') as data:
    all_images = data['training_images']
    all_labels = data['training_labels']

# plt.imshow(training_images[0].reshape(28, 28), cmap='gray')
# plt.show()
# for x in range(0, len(training_images)):

training_images = all_images[:int(len(all_images) / 2)]
training_labels = all_labels[:int(len(all_labels) / 2)]
testing_images = all_images[int(len(all_images) / 2):]
testing_labels = all_labels[int(len(all_images) / 2):]

layer_sizes = (784, 16, 16, 10)

net = nno.NeuralNetwork(layer_sizes)
net.print_accuracy(training_images, testing_labels)



# print prediction

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

for x in range(0, 1):
    a, b = unison_shuffled_copies(training_images, training_labels)
    net.update(50, a, b)
