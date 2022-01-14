import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Input, Layer
from tensorflow.keras.constraints import Constraint


class WeightClip(Constraint):
    """Clips the weights incident to each hidden unit to be inside a range"""
    
    def __init__(self, c=2):
        self.c = c
        
    def __call__(self, p):
        return tf.clip_by_value(p, clip_value_min=-self.c, clip_value_max=self.c)
    
    def get_config(self):
        return {'name': self.__class__.__name__, 'c': self.c}


class ModelBuilder(Model):
    """Neural network builder"""
    
    def __init__(self, 
                 input_shape: tuple, 
                 architecture: list[int] = [1, 3, 1],
                 weight_clipping: float = 7., 
                 activation: str = "tanh",
                 name: str = None, 
                 **kwargs
                ):
        """
        Initializes NN hyperparameters

        Args:
            n_input (tuple): shape of the input dataset
            architecture (list, optional): architecture of the neural network. Defaults to [1, 3, 1].
            weight_clipping (float, optional): weight clipping contraint. Defaults to 7.
            internal_activation (str, optional): activation function. Defaults to "tanh".
            custom_activation_bool (bool, optional): custom activation function. Defaults to False.
            custom_const (float, optional): custom activation function parameter. Defaults to 1.
            name (str, optional): name of the model. Defaults to None.
        """
        super().__init__(name=name, **kwargs)

        self.inputs = Input(
            shape=input_shape
        )
        
        self.hidden_layers = [
            Dense(
                    architecture[i+1], 
                    input_shape=(architecture[i],), 
                    activation=activation,
                    kernel_constraint = WeightClip(weight_clipping)
                ) 
            for i in range(len(architecture)-2)
        ]
        
        self.output_layer = Dense(
            architecture[-1],
            input_shape=(architecture[-2],), 
            activation="linear",
            kernel_constraint = WeightClip(weight_clipping)
        )
    
    
        def call(self, x):
            for hidden_layer in self.hidden_layers:
                x = hidden_layer(x)
            x = self.output_layer(x)
            return x
        
        