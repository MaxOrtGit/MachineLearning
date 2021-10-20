import numpy as np


class NeuralNetwork:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
    
        
        print(self.weights)
        print(self.biases)
        print(np.shape(self.weights))
        print(np.shape(self.biases))

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
            self.back_prop(np.average)
            




    def back_prop(self, cost):
        #print(self.weight_shapes)
        #print(self.weights[0])
        #print(np.shape(self.weights[0]))
        #print(self.biases[0])
        #print(np.shape(self.biases[0]))
        
        for row_on in range(len(self.sizes)-2, -1, -1):
            for cell_on in self.sizes[row_on]:
                zeighted_zeight = self.biases[row_on][cell_on]
                for weight_val in self.weights[row_on][cell_on]:
                    zeighted_zeight += weight_val   
                weight_cost = self.activation_derivative(zeighted_zeight)
                #I DOnt know, ill figure it out later
            
            



    @staticmethod
    def get_cost(predictions, labels):
        return np.square(np.subtract(predictions, labels))


    @staticmethod
    def get_gradient(weights):
        return np.gradient(np.concatenate(weights))

    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def activation_derivative(x):
        return (1 / (1 + np.exp(-x)))*(1-(1 / (1 + np.exp(-x))))
