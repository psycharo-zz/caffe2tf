#!/usr/bin/env python

## Uncomment this if necessary
# caffe_root = '<path-to-your-caffe>'
# import sys
# sys.path.append(caffe_root)
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import argparse
import pickle

def format_convolution(layer):
  fmt = "{0} = layers.conv2d({1}, {2}, {3}, {4}, activation_fn=None, scope='{5}')"
  param = layer.convolution_param
  return fmt.format(layer.top[0],
                    layer.bottom[0],
                    param.num_output,
                    param.kernel_size[0],
                    1,
                    layer.top[0])

def format_relu(layer):
  return "{0} = tf.nn.relu({1})".format(layer.top[0], layer.bottom[0])

def format_pooling(layer):
  param = layer.pooling_param
  pool = 'layers.max_pool2d' if param.pool == layer.pooling_param.MAX else 'layers.avg_pool2d'
  return "{0} = {1}({2}, {3}, {4})".format(layer.top[0],
                                           pool,
                                           layer.bottom[0],
                                           param.kernel_size,
                                           param.stride)

def format_concat(layer):
  return "{0} = tf.concat({1}, [{2}])".format(layer.top[0],
                                              convert_axis(layer.concat_param.axis),
                                              ', '.join(layer.bottom))

def format_input(net):
  N,C,H,W = net.input_dim
  assert len(net.input) == 1
  return "{0} = tf.placeholder(tf.float32, {1})".format(net.input[0], [N,H,W,C])

def format_network(net_pb, init_dict=None):
  src = ''
  src += format_input(net_pb) + '\n'
  for lid, layer in enumerate(net_pb.layer):
    if layer.type == 'Convolution':
      src += format_convolution(layer)
    elif layer.type == 'ReLU':
      src += format_relu(layer)
    elif layer.type == 'Pooling':
      src += format_pooling(layer)
    elif layer.type == 'Concat':
      src += format_concat(layer)
    else:
      raise ValueError('Unknown layer type: %s' % layer.type)
    src += '\n'
  return src

def convert_axis(axis):
  return {0:0, 1:3, 2:1, 3:2}[axis]

def convert_weights(weights):
  return weights.transpose((2,3,1,0)) if len(weights.shape) == 4 else weights

parser = argparse.ArgumentParser(description='Very simplistic converter of caffe models')
parser.add_argument('--model', dest='model_path', required=True, help='path to .prototxt')
parser.add_argument('--weights', dest='weights_path', required=True, help='path to .caffemodel')
parser.add_argument('--src', dest='src_path', required=True, help='path to save the resulting source code')
parser.add_argument('--pkl', dest='pkl_path', required=True, help='path to save the pickled parameters')

args = parser.parse_args()

## saving the weights in a dict
net = caffe.Net(args.model_path, caffe.TEST, weights=args.weights_path)
parameters = { name : [ convert_weights(blob.data) for blob in blobs ]
               for name, blobs in net.params.iteritems() }
pickle.dump(parameters, open(args.src_path, 'w'))

## saving the generated python code
net_pb = caffe_pb2.NetParameter()
net_pb = text_format.Parse(open(args.model_path).read(), net_pb)

src = format_network(net_pb)
with open(args.src_path, 'w') as f:
  f.writelines(src)

  
