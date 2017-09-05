import tensorflow as tf
import numpy as np
import random
import math

import matplotlib.pyplot as plt

# parmeters
T_in      = 10
#T_pred    = 20 - T_in
T_pred    = 1
BATCH     = 64 
#data     = tf.zeros( [T, BATCH, 32, 32, 128] )
num_steps = 1000
K         = 10 # number of frames used in future prediction
std_dev   = 0.1


def conv2d( x, W, b, strides = 1 ):
  x = tf.nn.conv2d(x, W, strides = [ 1, strides, strides, 1], padding = 'SAME')
  x = tf.nn.bias_add(x,b)
  return tf.nn.relu(x)

def fc( x, W, b):
  x = tf.matmul(x, W) + b
  return tf.nn.relu(x)


class LSTMAutoEncoder(object):
  def __init__(self, **kwargs):

    self.weights = {
        'wc1':  tf.Variable( tf.random_normal( [5, 5,  1, 24], stddev= std_dev ) ),
        'wc2':  tf.Variable( tf.random_normal( [5, 5, 24, 64], stddev= std_dev ) ),
        'wc3':  tf.Variable( tf.random_normal( [5, 5, 64, 64], stddev= std_dev ) ),
        'wfc1': tf.Variable( tf.random_normal( [1024, 4096],   stddev= std_dev) )  
    }
    
    self.biases = {
        'bc1' : tf.Variable( tf.random_normal([24],  stddev = 0) ),
        'bc2' : tf.Variable( tf.random_normal([64],  stddev = 0) ),
        'bc3' : tf.Variable( tf.random_normal([64],  stddev = 0) ), 
        'bfc1': tf.Variable( tf.random_normal([4096],stddev = 0) )
    }
    
  
  def EncoderDecoder(self, data ):
    
    # Encoder Decoder works over 2*T timesteps
    # First phase: Encoding
    # Put in real input data, discard output, keep state
    # Second phase: Decoding
    # Put in zero data (padding), use output/state

    with tf.variable_scope( "LSTM" ) as scope:
    
      lstm  = tf.contrib.rnn.LSTMCell( num_units = 1024 ) 
      state = lstm.zero_state( BATCH, "float" ) 
      datum = tf.split(data, T_in, axis = 1) # datum order: batch_size, T, height*width*channel

      # run lstm for T time step
      for t in range(T_in):
        if t > 0: 
            scope.reuse_variables()  # vary important! 
        
        output, state = lstm( tf.reshape( datum[t], [BATCH, -1] ), state) # inputs: 2-D tensor with shape [batch_size x input_size], state

      # what is tmp? datum at frame 0, why need to reshape it?
      tmp   = tf.reshape( datum[0], [BATCH, -1] )

      # Decoding phase
      zero_ = tf.zeros_like( tmp, "float" ) # generate a zero array using the shape of tmp

      output_list = []
      for t in range(T_pred):
        scope.reuse_variables() # this is important!      

        output, state = lstm( zero_, state ) # what is state here? 
        #output_list.append( tf.reshape( output, [BATCH,1,-1] ) ) # modify! 
        output_list.append( output )
      
      #print "output_list len", output_list.__len__()
      out = tf.concat( output_list, axis = 1 ) # ?
      return tf.reshape( out, [BATCH*T_pred, -1] )

  def load_validation(self):
    file = np.load( '/home/fensi/nas/Moving-MNIST/moving-mnist-valid.npz' )
    print file.keys()
    # ['clips', 'dims', 'input_raw_data']
    data = file['input_raw_data']
    return data

  def load_training(self):
    file = np.load( '/home/fensi/nas/Moving-MNIST/moving-mnist-train.npz' )
    # ['clips', 'dims', 'input_raw_data']
    data = file['input_raw_data'] # (200K, 1, 64, 64) --> (10K, 20, 64, 64)-> number of sequences, frames/sequence, height, width 
    data = np.reshape( data, [-1, 20, 64, 64, 1] )
     
    print "loading training data: data.shape", data.shape
    
    input_seq = data[:, 0:10]
    output_seq = data[:, 10:]
    
    return input_seq, output_seq

  def load_testing(self):
    file = np.load( '/home/fensi/nas/Moving-MNIST/moving-mnist-test.npz' )
    # ['clips', 'dims', 'input_raw_data']
    data = file['input_raw_data']
    data = np.reshape ( data, [-1, 20, 64, 64, 1] )
    input_seq = data [ :, 0:10]
    output_seq = data [ :, 10:] 
    return input_seq, output_seq

# Data
net             = LSTMAutoEncoder()
inp,out         = net.load_training()
tst_in, tst_out = net.load_testing() 

# Shuffle the sequence
perm            = range( inp.shape[0] ) # inp.shape[0] : 10k
random.shuffle( perm )


# Construct model
X               = tf.placeholder( "float", [BATCH, T_in,   64, 64, 1] ) # in TensorFlow, channel has to be at the last place
Y               = tf.placeholder( "float", [BATCH, T_pred, 64, 64, 1] ) 

# Conv2D with stride 2 (3 times)
# Reshape tensor to 4D (Conv2d only supports 4D tensors not 5D)
X_shape         = tf.reshape( X, [BATCH*T_in, 64, 64, 1] )
Y_shape         = tf.reshape( Y, [BATCH, T_pred, -1] )  # for computing loss, reshape as 3-D tensor

conv1           = conv2d( X_shape,   net.weights['wc1'], net.biases['bc1'], 2 )
conv2           = conv2d( conv1,     net.weights['wc2'], net.biases['bc2'], 2 )
conv3           = conv2d( conv2,     net.weights['wc3'], net.biases['bc3'], 2 )

# Reshape tensor to [BATCH, T, DIM]
res             = tf.reshape( conv3, [BATCH, T_in, -1] )

# LSTM encoders/decoders
# Input T_in frames, Output T_pred frames
prediction      = net.EncoderDecoder(res)
#exit(0)
print "prediction", prediction.get_shape().as_list() # prediction [T * BATCH, 1024] = 10x BATCH (number of sequence you process in parallel, the output is one feature for each frame )

# Fully connected: (BATCH*10, 1024) -->(BATCH*10, 4096) 
fc_out          = fc( prediction, net.weights['wfc1'], net.biases['bfc1'] )
print "fc_out shape", fc_out.shape


# Reshape fc_out to BATCH x T x DIM
fc_out          = tf.reshape( fc_out, [BATCH, T_pred, -1] )
sig_out         = tf.sigmoid( fc_out )


# define loss and optimizer
#loss_op         = tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits( logits = fc_out, labels = Y_shape ), axis = 1 ) # batch x 10 --> batch x 1
#loss_op         = tf.reduce_sum( loss_op, axis=1 )
#loss_op         = tf.reduce_mean( loss_op, axis=0 ) # loss is averaged over the batch  

diff            = fc_out - Y_shape
loss_op         = 0.5 * tf.reduce_sum( tf.reduce_sum( diff * diff, axis = 2 ), axis = 1)
loss_op         = tf.reduce_mean( loss_op )
train_op        = tf.train.AdamOptimizer( learning_rate = 0.001 ).minimize(loss_op)


print "inp.shape", inp.shape

with tf.Session() as sess:
  init            = tf.global_variables_initializer()
  sess.run( init )
  
  # Number of Epochs
  for e in range( 50 ):
    # Iterations
    for start in range( 0, inp.shape[0]-(inp.shape[0]%BATCH), BATCH  ):
      batch_x = np.zeros( [BATCH, T_in,   64, 64, 1] )
      batch_y = np.zeros( [BATCH, T_pred, 64, 64, 1] )

      for b in range( BATCH ):
        batch_x[b] = inp[ perm[ start+b ], : T_in ]
        batch_y[b] = out[ perm[ start+b ], : T_pred ]

      feed = { X: batch_x, Y: batch_y }
      op, loss, p1 = sess.run([train_op, loss_op, prediction], feed_dict = feed )
      
      print loss
  
  # Testing the reconstruction 
  batch_x      = inp[ 0 : BATCH ]
  img_pre, img = sess.run( [fc_out, sig_out], feed_dict = { X : batch_x } )
  
  img_pre      = np.reshape( img_pre, [BATCH, T_pred, 64, 64] )
  img          = np.reshape( img, [BATCH, T_pred, 64, 64] )
  
  for t in range( T_pred ):
    plt.imshow( img_pre[0,t] )
    plt.show()
