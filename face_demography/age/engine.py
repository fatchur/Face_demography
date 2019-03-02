import os
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import glob

# class info
classes = np.array(['11-20','21-30', '31-40','41-50','51-60','61-'])
num_classes = len(classes)
temp = classes.tolist()

def create_graph(model_path):
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name = '')
        print ('selesai')

def extract_features(image_path, verbose = False):
   
    with tf.Session() as sess:
        flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        image_data = gfile.FastGFile(image_path, 'rb').read()
        feature = sess.run(flattened_tensor, {
                'DecodeJpeg/contents:0': image_data
        })      
    return feature

path = 'inception_dec_2015/tensorflow_inception_graph.pb'
create_graph(path)

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_fc_layer(input,
                 num_inputs,
                 num_outputs,
                 use_relu = True):
    weights = new_weights(shape = [num_inputs, num_outputs])
    biases = new_biases(length = num_outputs)
    
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
        
    return layer


feature_size_flat = 2048
dropout = 0.75

x = tf.placeholder(tf.float32, shape=[None, feature_size_flat], name = 'x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name = 'y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_fc1 = new_fc_layer(input = x,
                        num_inputs = 2048,
                        num_outputs = 1000,
                        use_relu = True)
#layer_fc1 = tf.nn.dropout(layer_fc1, dropout)

layer_fc2 = new_fc_layer(input = layer_fc1,
                        num_inputs = 1000,
                        num_outputs = 1000,
                        use_relu = True)
layer_fc2 = tf.nn.dropout(layer_fc2, dropout)

layer_fc3 = new_fc_layer(input = layer_fc2,
                        num_inputs = 1000,
                        num_outputs = 1000,
                        use_relu = True)
layer_fc3 = tf.nn.dropout(layer_fc3, dropout)

layer_fc4 = new_fc_layer(input = layer_fc3,
                        num_inputs = 1000,
                        num_outputs = 1000,
                        use_relu = True)
layer_fc4 = tf.nn.dropout(layer_fc4, dropout)

layer_fc5 = new_fc_layer(input = layer_fc4,
                        num_inputs = 1000,
                        num_outputs = 6,
                        use_relu = True)
layer_fc5 = tf.nn.dropout(layer_fc5, dropout)

y_pred = tf.nn.softmax(layer_fc5)
y_pred_cls = tf.argmax(y_pred, dimension = 1)

saver = tf.train.Saver()
session = tf.Session()
save_path = "age_transfer2/age_transfer2"
saver.restore(sess= session, save_path = save_path)

image_path = 'age_testing2/11-20/15-20-3321.jpg'
images = extract_features(image_path, verbose = False) 

transfer_values = np.asarray(images)
transfer_values = transfer_values.reshape((1, 2048))

feed_dict_test = {x: transfer_values}
prediction_test = session.run(y_pred_cls, feed_dict = feed_dict_test)
print prediction_test





