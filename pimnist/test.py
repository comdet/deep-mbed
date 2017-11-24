import mxnet as mx
import numpy as np
import cv2,sys,time
from collections import namedtuple

def loadModel(modelname):
    t1 = time.time()
    sym, arg_params, aux_params = mx.model.load_checkpoint(modelname, 0)
    t2 = time.time()
    t = 1000*(t2-t1)
    print("Loaded in %2.2f microseconds" % t)
    arg_params['prob_label'] = mx.nd.array([0])
    mod = mx.mod.Module(symbol=sym, context=mx.cpu())
    mod.bind(for_training=False, data_shapes=[('data', (1,1,28,28))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params)
    return mod

def prepareNDArray(img):
    #img = np.swapaxes(img, 0, 2)
    #img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis,np.newaxis, :]
    #print(img.shape)
    return mx.nd.array(img)
      
st = 0
def tstart():
    global st
    st = time.time()

def tstop():
    global st
    return 1000*(time.time() - st)

def predict(filename, model):
    array = prepareNDArray(filename)
    Batch = namedtuple('Batch', ['data'])
    tstart()
    model.forward(Batch([array]))
    prob = model.get_outputs()[0]
    prob.wait_to_read()
    #print("Predicted in %2.2f microseconds" % tstop())
    prob = prob[0].asnumpy()
    prob = np.squeeze(prob)
    num = np.argmax(prob)
    return (num,prob[num],tstop())

def init(modelname):
    model = loadModel(modelname)    
    return model

lenet = init("lenet")

def predict_local(filename):
    res = predict(filename,lenet)
    print("Result time(%0.2f)[microsec] con (%f) : %d " %(res[2],res[1],res[0]))
    return res
#print ("*** Inception v3")
#print predict(filename,inceptionv3,c,5)
