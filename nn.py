import numpy as np

def affine_forward(W,b,x):
    # compute fwd pass, return cache (W,b,x)
    # W: (D,C)
    # b: (C)
    # x: (N,D)
    out = x@W+b
    cache = W,b,x
    return out, cache

def affine_backward(dout, cache):
    # returns gradients (dW, db, dx)

    W,b,x = cache
    # dout: (N,C), since it's the same size as output of affine_forward

    dW = x.T @ dout # (D,C)
    db = np.ones_like(b)
    dx = dout@W.T # (N,D)

    return dW,db,dx

def relu_forward():
    pass

def relu_backward():
    pass

def softmax_forward():
    pass

def softmax_backward():
    pass

class FCNet:
    # architecture: {affine - relu}*(L-1) - affine - softmax
    def __init__(
            self,
            hidden_dims, # list of hidden dimensions
            input_dim=28*28,
            num_classes=10,
            reg=0.0, # l2 reg
            weight_scale=1e-2, # initial weight scale
            dtype=np.float32
    ):
        # stores weights and biases (in the form W#, b#)
        # self.params['W1'] are first layer weights
        self.params = dict()

        prev_dim = input_dim
        for i in range(len(hidden_dims)):
            self.params[f'W{i+1}'] = weight_scale*np.random.randn(prev_dim, hidden_dims[i]).astype(dtype)
            self.params[f'b{i+1}'] = np.zeros(hidden_dims[i]).astype(dtype)
            prev_dim = hidden_dims[i]

        # need one more layer for softmax
        self.params[f'W{i}'] = weight_scale*np.random.randn(prev_dim, num_classes).astype(dtype)
        self.params[f'b{i}'] = np.zeros().astype(dtype)

    def loss(self, X, y=None):
        '''
        Returns loss on batch X with labels y.
        If y is None, returns raw scores.
        '''
        pass

# optimizers
# note optimizers update in-place

def sgd(w, dw, lr=1e-2):
    # config is dict containing learning_rate
    # updates 
    w -= lr * dw
    return w

class Solver:
    def __init__(
        self,
        X, y,
        batch_size=250,
        num_iter=1500,
        reg=1e-1,
        optimizer=sgd
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_iter = num_iter
        self.reg = reg
        self.optimizer = optimizer

    def train(self):
        pass

