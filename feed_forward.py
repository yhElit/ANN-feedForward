import pandas as pd
import random


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.num_inputs = len(weights)

    @staticmethod
    def relu(value):
        if value < 0:
            return 0

        return value

    def predict(self, values):
        assert self.num_inputs == len(values)
        result = 0

        for i in range(self.num_inputs):
            result = result + values[i] * self.weights[i]

        return Neuron.relu(result + self.bias)

    def __str__(self):
        return f"(weights: {self.weights} - bias: {self.bias})"


class ForwardNeuralNetwork:
    def __init__(self):
        self.hidden_layers = list()
        self.output_layer = list()

    @staticmethod
    def build_neuron(weight_count):
        weights = list()

        for i in range(weight_count):
            weights.append(random.uniform(0, 0.5))

        bias = random.uniform(0, 0.5)
        return Neuron(weights=weights, bias=bias)

    def build_layers(self, x_train):
        weight_count = x_train.shape[1]

        self.hidden_layers.append([ForwardNeuralNetwork.build_neuron(weight_count), ForwardNeuralNetwork.build_neuron(weight_count), ForwardNeuralNetwork.build_neuron(weight_count)])

        next_layer_weight_count = len(self.hidden_layers[0])

        self.hidden_layers.append([ForwardNeuralNetwork.build_neuron(next_layer_weight_count), ForwardNeuralNetwork.build_neuron(next_layer_weight_count), ForwardNeuralNetwork.build_neuron(next_layer_weight_count), ForwardNeuralNetwork.build_neuron(next_layer_weight_count)])

        next_layer_weight_count = len(self.hidden_layers[1])

        self.output_layer.append(ForwardNeuralNetwork.build_neuron(next_layer_weight_count))

    def fit(self, x_train):
        self.build_layers(x_train)

    def predict(self, x_test):
        result = list()

        for index, row in x_test.iterrows():
            feed_to_next_layer = row.to_list()

            for layer_index in range(len(self.hidden_layers)):
                layer_values = list()

                for neuron in self.hidden_layers[layer_index]:
                    neuron_prediction = neuron.predict(feed_to_next_layer)
                    layer_values.append(neuron_prediction)

                feed_to_next_layer = layer_values

            result.append(self.output_layer[0].predict(feed_to_next_layer))

        return result


def main():
    features = ["feature_1", "feature_2"]
    label = ["class"]
    columns = features + label

    samples = list()
    samples.append([0.8888, 0.7777, 0.4444])

    data = pd.DataFrame(samples, columns=columns)

    model = ForwardNeuralNetwork()
    model.fit(data[features])

    predictions = model.predict(data[features])
    prediction = predictions[0]

    print("prediction: ", prediction)
    actual = data.iloc[0, 2]

    print("actual: ", actual)
    print("loss: ", actual - prediction)


if __name__ == '__main__':
    main()
