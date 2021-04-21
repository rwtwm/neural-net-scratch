import structure as stc
import numpy as np

# The layer is the area between the 'neurons' between each column.
# The neurons themselves aren't modelled directly, but the values
# from each neuron are transported between these layers.
class Layer:
    def __init__(self, ins, outs, act_function):
        self.ins = ins
        self.outs = outs
        self.act_function = act_function

        # The underscore indicates that these are private variables.
        self._W = stc.create_weight_matrix(self.outs, self.ins)
        self._b = stc.create_bias_vector(self.outs)

    def forward_pass(self, x):
        return self.act_function(np.dot(self._W, x) + self._b)


if __name__== "__main__":
    l1 = Layer(2, 4, stc.leaky_relu)
    l2 = Layer(4, 4, stc.leaky_relu)
    l3 = Layer(4, 1, stc.leaky_relu)
    x = stc.get_scaled_normal(2)
    p1 = l1.forward_pass(x)
    p2 = l2.forward_pass(p1)
    p3 = l3.forward_pass(p2)
    print(p3)

    