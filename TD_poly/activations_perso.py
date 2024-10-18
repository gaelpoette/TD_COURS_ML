import tensorflow as tf
#import tf.keras as K

def polyTwo(z):
    return tf.pow(z,2)  - 1

def polyThree(z):
    return 2*tf.pow(z,3)-3*tf.pow(z,2)+5

def polyFive(z):
    return tf.pow(z,5)-4*tf.pow(z,4)+2*tf.pow(z,3)+8*tf.pow(z,2)-11*z-12
