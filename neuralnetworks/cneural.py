from clayer import Layer
import structure as stc


class NeuralNetwork:
    '''A series of connected layers'''
    def __init__(self, layers: list[Layer], input=None, loss_func=None):
        self._layers = layers
        self._input = input
        self._loss = loss_func

        # Check the layers are compatible
        # Dual iteration. Zip the results of two iterators in a tuple
        # But unpack them in the reference variables.
        for (current, next) in zip(layers[:-1], layers[1:]):
            if current.outs != next.ins:
                raise ValueError("Layer shapes are incompatible")

    def add_input(self, x=None):
        '''Adds a default input for testing in absence of a specific input'''
        input_size = self._layers[0].ins
        if x is None:
            x = stc.get_scaled_normal(input_size)
        self._input = x

    def add_loss(self, loss_function):
        self._loss = loss_function

    def run_loss(self, values, expected):
        return self._loss(values, expected)

    def forward_propagate(self):
        '''Combines the forward passes of all of the layers in the network.'''
        out = self._input
        for layer in self._layers:
            out = layer.forward_pass(out)
        return out



if __name__ == "__main__":
    l1 = Layer(2, 4, stc.leaky_relu)
    l2 = Layer(4, 4, stc.leaky_relu)
    l3 = Layer(4, 1, stc.leaky_relu)
    net = NeuralNetwork([l1, l2, l3])
    net.add_input()
    output = net.forward_propagate()
    print(output)
