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
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params)
    return mod

def loadCategories():
    synsetfile = open('synset.txt', 'r')
    synsets = []
    for l in synsetfile:
            synsets.append(l.rstrip())
    return synsets

def prepareNDArray(img):
    #img = cv2.imread(filename)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(img, (224, 224,))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return mx.nd.array(img)
      
st = 0
def tstart():
    global st
    st = time.time()

def tstop():
    global st
    return 1000*(time.time() - st)

def predict(filename, model, categories, n):
    array = prepareNDArray(filename)
    Batch = namedtuple('Batch', ['data'])
    tstart()
    model.forward(Batch([array]))
    prob = model.get_outputs()[0]
    prob.wait_to_read()
    #print("Predicted in %2.2f microseconds" % tstop())
    prob = prob[0].asnumpy()
    prob = np.squeeze(prob)
    sortedprobindex = np.argsort(prob)[::-1]
    topn = []
    for i in sortedprobindex[0:n]:
            topn.append((prob[i], categories[i],tstop()))
    return topn

def init(modelname):
    model = loadModel(modelname)
    cats = loadCategories()
    return model, cats

inceptionv3,c = init("Inception-BN")

def predict_local(filename):
    res = predict(filename,inceptionv3,c,5)
    print("Result (%0.4f) time(%0.2f): %s " %(res[0][0],res[0][2],res[0][1]))
#print ("*** Inception v3")
#print predict(filename,inceptionv3,c,5)
