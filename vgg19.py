# VGG-19, 19-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
# License: non-commercial use only

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer


#from lasagne.layers.conv import Conv2DLayer as ConvLayer
#from lasagne.layers import Pool2DLayer as PoolLayer

from lasagne.layers.corrmm import Conv2DMMLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer

#from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayer

from lasagne.nonlinearities import softmax

from collections import OrderedDict

def build_model(pool_mode='max'):
    net = OrderedDict() 
    net['input'] = InputLayer((None, 3, 224, 224), name='input')
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, name='conv1_1')
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, name="conv1_2")
    net['pool1'] = PoolLayer(net['conv1_2'], 2, name="pool1", mode=pool_mode)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, name="conv2_1")
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, name="conv2_2")
    net['pool2'] = PoolLayer(net['conv2_2'], 2, name="pool2", mode=pool_mode)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, name="conv3_1")
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, name="conv3_2")
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, name="conv3_3")
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1, name="conv3_4")
    net['pool3'] = PoolLayer(net['conv3_4'], 2, name="pool3", mode=pool_mode)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, name="conv4_1")
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, name="conv4_2")
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, name="conv4_3")
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1, name="conv4_4")
    net['pool4'] = PoolLayer(net['conv4_4'], 2, name="pool4", mode=pool_mode)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, name="conv5_1")
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, name="conv5_2")
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, name="conv5_3")
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1, name="conv5_4")
    net['pool5'] = PoolLayer(net['conv5_4'], 2, name="pool5", mode=pool_mode)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096, name="fc6")
    net['fc7'] = DenseLayer(net['fc6'], num_units=4096, name="fc7")
    net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None, name="fc8")
    net['prob'] = NonlinearityLayer(net['fc8'], softmax, name="prob")

    return net

if __name__ == "__main__":
    import cPickle as pickle
    model = build_model()

    data = pickle.load(open("vgg19.pkl"))
    
    weights = data["param values"]
    mean_value = data["mean value"]
    classes = data["synset words"]


    i = 0
    for k in model.keys():
        if hasattr(model[k], "W") and hasattr(model[k], "b"):
            model[k].W.set_value(weights[i])
            model[k].b.set_value(weights[i + 1])
            print(model[k].W.get_value().shape, weights[i].shape)
            print(model[k].b.get_value().shape, weights[i +1].shape)
            i += 2
    
    fd = open("vgg19-lasagne.pkl", "w")
    data = dict(model=model, classes=classes, mean_value=mean_value)
    pickle.dump(data, fd)
    fd.close()
