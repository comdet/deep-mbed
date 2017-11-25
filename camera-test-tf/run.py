#!bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import re
import sys
import tarfile
import time
import numpy as np
from six.moves import urllib
import tensorflow as tf
import picamera
import picamera.array
import cv2

FLAGS = tf.app.flags.FLAGS

model_dir='/tmp/imagenet'
num_top_predictions=1
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

camera = picamera.PiCamera()
rawCapture = picamera.array.PiRGBArray(camera)

class NodeLookup(object):
    def __init__(self,label_lookup_path=None,uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(Fmodel_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):    
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def create_graph():
    with tf.gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()

      # Creates graph from saved GraphDef.
    start_time = time.time()
    create_graph()
    graph_time = time.time() - start_time
    print('Build graph time: %0.4f' % graph_time)

    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        node_lookup = NodeLookup()
        # MODIFICATION BY SAM ABRAHAMS
        while True:
            camera.capture(rawCapture,format="rgb")
            img = rawCapture.array
            img = cv2.resize(img, (224, 224,))
            cv2.imshow("Image",img)

            start_time = time.time()    
            predictions = sess.run(softmax_tensor,
                                 {'DecodeJpeg/contents:0': img})
            running_time = time.time() - start_time
            predictions = np.squeeze(predictions)
            top_k = predictions.argsort()[-num_top_predictions:][::-1]
            
            #for node_id in top_k:
            #    human_string = node_lookup.id_to_string(node_id)
            #    score = predictions[node_id]
            #    print('Test time : %.3f %s (score = %.5f)' % (running_time,human_string, score))
            human_string = node_lookup.id_to_string(top_k[0])
            print('Test time %.3f result= %s (score = %.5f)' % (running_time,human_string,score))
            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)

def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
    maybe_download_and_extract()
    run_inference_on_image()

if __name__ == '__main__':
    tf.app.run()
