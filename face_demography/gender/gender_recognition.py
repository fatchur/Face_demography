import cv2
import tensorflow as tf
import numpy as np
import math

#num of image channel
num_channels = 3
#image size
img_size = 90
# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)
#convolutional layer 1

filter_size1 = 10
num_filters1 = 64
#convolutional layer 2
filter_size2 = 5
num_filters2 = 64
#convolutional layer 3
filter_size3 = 5
num_filters3 = 32
#fully connected layer
fc_size = 128

# class info
classes = np.array(["MALE", "FEMALE"])
num_classes = len(classes)

save_path = 'gender_model3/gender_model3'

cascPath = 'haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
image_size = 90
video_capture = cv2.VideoCapture(0)


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

a = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='a')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
x_image = tf.reshape(a, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                 num_input_channels=num_channels,
                 filter_size=filter_size1,
                 num_filters=num_filters1,
                 use_pooling=True)

layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                 num_input_channels=num_filters1,
                 filter_size=filter_size2,
                 num_filters=num_filters2,
                 use_pooling=True)

layer_conv3, weights_conv3 = \
    new_conv_layer(input=layer_conv2,
                 num_input_channels=num_filters2,
                 filter_size=filter_size3,
                 num_filters=num_filters3,
                 use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv3)
layer_fc1 = new_fc_layer(input=layer_flat,
                      num_inputs=num_features,
                      num_outputs=fc_size,
                      use_relu=True)
layer_fc2 = new_fc_layer(input=layer_fc1,
                      num_inputs=fc_size,
                      num_outputs=num_classes,
                      use_relu=False)
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)
	
saver = tf.train.Saver()
session = tf.Session()
saver.restore(sess=session, save_path=save_path)
    
   	
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(90, 90)
        #flags=cv2.CV_HAAR_SCALE_IMAGE
    )
  
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image = frame[y:(y+h), x:(x+h)]
        img = cv2.resize(image,(int(90), int(90)))
        input = img.reshape((1,24300))

    	feed_dict = {a: input}
	    peluang = y_pred.eval(feed_dict, session = session)
    	hasil = y_pred_cls.eval(feed_dict, session = session)
        ss = classes[hasil]
        cv2.putText(frame, str(ss) ,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)
        
	cv2.putText(frame, str(peluang) ,(x,y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)
	session.close()

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


