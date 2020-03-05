import numpy as np
import os
import matplotlib.pyplot as plt
import time
import csv
import timeit
from PIL import Image
from skimage import transform, io
import tensorflow as tf

import numpy
import math
import glob
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#fi = open('test.csv', 'w', newline='')
#wr = csv.writer(fi)
#wr.writerow(['Filename', 'class', 'Label', 'Real_label', 'True/False'])

label_ck=list()
flag = 0

### datalist ////   0 == train   //   1 == test

datalist = ["elf_image/"]

for z in range(1):
    DATAPATH = datalist[z]

### SPECIFY THE FOLDER PATHS
### + RESHAPE SIZE + GRAYSCALE
    
# Training set folder

    paths = set(os.listdir(DATAPATH))

# The reshape size
    imgsize = [64, 64]

# Grayscale
    use_gray = 1

# Save name
    data_name = "add_normal"

# Path alarm
    for i, path in enumerate(paths) :
        #print (" [%d/%d] %s" % (i+1, len(paths), DATAPATH + path))
        if flag<4:
            label_ck.append(path)
        flag+=1
    #print(label_ck)

### RGB 2 GRAY FUNCTION
    def rgb2gray(rgb) :
        if len(rgb.shape) is 3 :
            return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        else :
            return rgb # Current Image gray

### LOAD IMAGES
    nclass     = len(paths)
    valid_exts = [".jpg",".gif",".png",".tga", ".jpeg"]
    errcnt     = 0
    imgcnt     = 0

    fname=list()
    for i, relpath in zip(range(nclass), paths) :
        
        path = DATAPATH + relpath
        '''
        flist = os.listdir(path)
        print(path)
        '''
        if os.path.splitext(relpath)[1].lower() not in valid_exts:
            errcnt = errcnt + 1
            continue
        
        currimg  = io.imread(path)


        # Convert to grayscale  
        if use_gray:
            grayimg  = rgb2gray(currimg)
        else:
            grayimg  = currimg

        # Reshape
        graysmall = transform.resize(grayimg, [imgsize[0], imgsize[1]])/255.
        grayvec   = np.reshape(graysmall, (1, -1))

        # Save 
        curr_label = np.eye(1, 2)
        #print(curr_label)
        if imgcnt is 0:
            fname.append(relpath)
            totalimg   = grayvec
            totallabel = curr_label
            #print("total ====== ",totalimg)
            #print("total L ====== ",totallabel)
        else:
            fname.append(relpath)
            totalimg   = np.concatenate((totalimg, grayvec), axis=0)
            totallabel = np.concatenate((totallabel, curr_label), axis=0)
            #print("total ====== ",totalimg)
            #print("total L ====== ",totallabel)
        imgcnt = imgcnt + 1
        
    print ("%d images load error : FormatError" % (errcnt))
    print ("Total %d images loaded." % (imgcnt))

### DIVIDE TOTAL DATA INTO TRAINING AND TEST SET
#def print_shape(string, x):
#    print ("Shape of '%s' is %s" % (string, x.shape,))

    testfname = fname
    testimg    = totalimg
    testlabel  = totallabel

ntest  = testimg.shape[0]

'''
### DEFINE NETWORK
tf.set_random_seed(0)
n_input  = dim
n_output = nclass

'''

if use_gray :
    weights = {
        'wc1' : tf.Variable(tf.random.normal([3, 3, 1, 128], stddev=0.1)),
        'wc2' : tf.Variable(tf.random.normal([3, 3, 128, 128], stddev=0.1)),
        'wd1' : tf.Variable(tf.random.normal([(int)(imgsize[0]/4*imgsize[1]/4)*128, 128], stddev=0.1)),
        'wd2' : tf.Variable(tf.random.normal([128, 2], stddev=0.1))
    }
else :
    weights = {
        'wc1' : tf.Variable(tf.random.normal([3, 3, 3, 128], stddev=0.1)),
        'wc2' : tf.Variable(tf.random.normal([3, 3, 128, 128], stddev=0.1)),
        'wd1' : tf.Variable(tf.random.normal([(int)(imgsize[0]/4*imgsize[1]/4)*128, 128], stddev=0.1)),
        'wd2' : tf.Variable(tf.random.normal([128, 2], stddev=0.1))
    }

biases = {
    'bc1': tf.Variable(tf.random.normal([128], stddev=0.1)),
    'bc2': tf.Variable(tf.random.normal([128], stddev=0.1)),
    'bd1': tf.Variable(tf.random.normal([128], stddev=0.1)),
    'bd2': tf.Variable(tf.random.normal([2], stddev=0.1))
}



def chck(label):
    if label[0]==1:
        return 1
    elif label[1] == 1:
        return 2

def conv_basic(_input, _w, _b, _keepratio, _use_gray) :
    # Input
    if _use_gray:
        _input_r = tf.reshape(_input, shape=[-1, imgsize[0], imgsize[1], 1])
    else:
        _input_r = tf.reshape(_input, shape=[-1, imgsize[0], imgsize[1], 3])

    # Conv layer 1
    _conv1    = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME'), _b['bc1']))
    _pool1    = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)

    # Conv layer 2
    _conv2    = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME'), _b['bc2']))
    _pool2    = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)

    # Vecctorize
    _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])

    # Fully connected layer 1
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)

    # Fully connected layer 2
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])

    # Return
    out = {
        'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1
        , 'pool1_dr1': _pool_dr1, 'conv2': _conv2, 'pool2': _pool2
        , 'pool_dr2': _pool_dr2, 'dense1': _dense1, 'fc1': _fc1
        , 'fc_dr1': _fc_dr1, 'out': _out
    }
    return out



### DEFINE FUNCTIONS
# tf Graph input
x = tf.placeholder(tf.float32, [None, 64*64])
y = tf.placeholder(tf.float32, [None, 2])
keepratio = tf.placeholder(tf.float32)



# Funtions
_pred = conv_basic(x, weights, biases, keepratio, use_gray)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=_pred, labels=y))


WEIGHT_DECAY_FACTOR = 0.0001
l2_loss = tf.add_n([tf.nn.l2_loss(v) 
            for v in tf.trainable_variables()])
cost = cost + WEIGHT_DECAY_FACTOR*l2_loss
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
_corr = tf.equal(tf.argmax(_pred,1), tf.argmax(y,1)) # Count corrects
testL = tf.argmax(_pred,1)
test_label = tf.one_hot(testL, 2)



# 모델 불러오기
    
sess = tf.Session()

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

print("prediction!!")
log_save = './module/modelcheckpoint/CNN_Checkpoint-99.ckpt-99'
#저장한 모델 다시 불러오기 
saver.restore(sess, log_save)


fi = open('result.csv', 'w', newline='')
wr = csv.writer(fi)
wr.writerow(['Filename', 'Label'])

test_Label = sess.run(test_label, feed_dict={x: testimg, y: testlabel,keepratio:1.})
cast = tf.cast(y, tf.float32)
label = sess.run(cast, feed_dict={x : testimg, y : testlabel})      #정답지 
check = sess.run(_corr, feed_dict={x: testimg, y: testlabel, keepratio:1.})

for i in range(len(testfname)):
    wr.writerow([testfname[i], int(chck(test_Label[i])) % 2])

fi.close()
