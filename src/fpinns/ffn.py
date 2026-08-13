import torch
import torch.nn as nn

_activations_map = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
}


class Sine(nn.Module):
    def __init__(self, layer_size : list[int], freq):
        super().__init__()
        self.input_size, self.output_size = layer_size
        self.linear = nn.Linear(self.input_size,self.output_size)
        self.frequency = freq
    def forward(self, x):
        return torch.sin(self.frequency*self.linear(x))

# Mask
class mask_activation(nn.Module):
    def __init__(self, a: list) -> None:
        super().__init__()
        self.activation, size, initial_value = a
        self.learnable_vector = initial_value*nn.Parameter(torch.ones((1, size), dtype=torch.float32))

    def forward(self, x):
        return self.activation(x)*(1-torch.exp(-(self.learnable_vector.detach()*x)**2))


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_count, freq, activation_name, activation_init, use_mask) -> None:
        super().__init__()         # This initialises the parent class (nn.Module) based on self pointer of child class (Net), so that child class is compatible with pytorch operations 
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size
        if use_mask:
            activation = mask_activation([_activations_map[activation_name](), self.hidden_size, activation_init])
        else:
            activation = _activations_map[activation_name]()

        self.Ext_Layers = [Sine([self.input_size,self.hidden_size],freq)]

        self.Layer = nn.ModuleList([nn.Linear(self.hidden_size,self.hidden_size)])

        for _ in range(layer_count - 1):
            self.Layer.extend([nn.Linear(self.hidden_size, self.hidden_size), activation])
        self.Out_Layer = nn.Linear(self.hidden_size,self.output_size)

    def forward(self,x):
        for layers in self.Ext_Layers:
            x = layers(x)
        for layers in self.Layer:
            x = layers(x)
        return self.Out_Layer(x)
