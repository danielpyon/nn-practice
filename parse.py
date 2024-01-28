import numpy as np

# return a tuple: (Nx28x28), (N,)
def parse(path: str) -> np.array:
    with open(path, 'r') as f:
        imgs = [list(map(int, line.split(','))) for line in f.read().strip().split('\n')]

    N = len(imgs)

    data = np.zeros((N,28,28))
    labels = np.zeros((N))

    for idx in range(len(imgs)):
        img = imgs[idx]
        label, pixels = img[0], img[1:]
        assert len(pixels) == 784

        # set label
        labels[idx] = label

        for i in range(28):
            for j in range(28):
                data[idx][i,j] = pixels[i * 28 + j]
    return data, labels

