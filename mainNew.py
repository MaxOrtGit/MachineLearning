import NeuralNetworkOld as nno
import NetworkR as nn
import numpy as np
import matplotlib.pyplot as plt

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']

# plt.imshow(training_images[0].reshape(28, 28), cmap='gray')
# plt.show()

layer_sizes = (784, 16, 16, 10)

net = nn.Network(layer_sizes)
