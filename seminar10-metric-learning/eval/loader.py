import pickle
import numpy as np
# import mxnet as mx
from PIL import Image
import io


def load_bin(path, image_size):
    print(path)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data_list = []

    for flip in [0, 1]:
        data = np.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)

    i = 0

    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        # img = mx.image.imdecode(_bin).asnumpy()
        img = np.array(Image.open(io.BytesIO(_bin)))
        img = np.transpose(img, axes=(2, 0, 1))

        for flip in [0, 1]:
            if flip == 1:
                img = np.flip(img, axis=2)

            data_list[flip][i] = img

        i += 1

        if i % 1000 == 0:
            print('loading bin', i)

    print(data_list[0].shape)
    return (data_list, issame_list)
