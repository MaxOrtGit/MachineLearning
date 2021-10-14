import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        self.weight_shapes = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [np.random.standard_normal(s) / s[1] ** .5 for s in self.weight_shapes]

        self.biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]
        print(self.weight_shapes)
        print(self.weights)

    def predict(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.activation(np.matmul(w, a) + b)
        return a

    def print_accuracy(self, images, labels):
        predictions = self.predict(images)
        num_correct = sum([np.argmax(a) == np.argmax(b) for a, b in zip(predictions, labels)])
        print('{0}/{1} accuracy: {2}%'.format(num_correct, len(images), (num_correct / len(images)) * 100))

    def update(self, step_size, change_by, images, labels):
        print(step_size)
        print(len(images))
        # print(images[1])


        for batchNumber in range(0, int(np.ceil(len(images) / step_size))):
            leRange = range(batchNumber * step_size,
                            ((batchNumber + 1) * step_size) if ((batchNumber + 1) * step_size <= len(images)) else (
                                    len(images) - 1))

            print(leRange)
            images_in_set = images[leRange]
            labels_in_set = labels[leRange]
            print(len(images_in_set))
            predictions = self.predict(images_in_set)
            print("fsdfdf")
            print(np.shape(predictions))
            costs_for_math = self.get_cost(predictions, labels_in_set)
            average = [np.average(k) for k in costs_for_math]
            #self.back_prop(np.average, 0)




   # def back_prop(self, cost, shape_index):
        #for()



    @staticmethod
    def get_cost(predictions, labels):
        return np.square(np.subtract(predictions, labels))


    @staticmethod
    def get_gradient(weights):
        return np.gradient(np.concatenate(weights))

    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))
