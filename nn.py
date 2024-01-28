import numpy as np

def affine_forward():
    pass

def affine_backward():
    pass

def relu_forward():
    pass

def relu_backward():
    pass

def softmax_forward():
    pass

def softmax_backward():
    pass

class FCNet:
    # {affine - relu}*(L-1) - affine - softmax
    def __init__(
            self,
            hidden_dims,
            input_dim=28*28,
            num_classes=10,
            reg=0.0,
            weight_scale=1e-2,
            dtype=np.float32
    ):
        # initialize weights
        pass

    def loss(self, X, y=None):
        '''
        Returns loss on batch X with labels y.
        If y is None, returns raw scores.
        '''
        pass


