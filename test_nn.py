def numeric_grad(M, f):
    pass

def affine_gradcheck(self):
    
    W = np.random.random((5,10))
    b = np.random.random(10)
    x = np.random.random((3,5))

    out, cache = affine_forward(W,b,x)
    dW, db, dx = affine_backward(1.0, cache)

    # TODO: implement gradcheck
    assert dW

if __name__ == '__main__':
    affine_gradcheck()
