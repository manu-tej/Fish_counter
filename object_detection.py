#!/usr/bin/env python
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pandas as pd
import time
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from skimage.draw import circle_perimeter, circle, line,line_aa
import deeplabcut
import shutil
import math

import logging
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
try:
    font = ImageFont.truetype('LiberationMono-Bold.ttf', 24)
except IOError:
    font = ImageFont.load_default()

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

colorclass=plt.cm.ScalarMappable(cmap='jet')
C=colorclass.to_rgba(np.linspace(0,1,4))
colors=(C[:,:3]*255).astype(np.uint8)

col_part = {'Mouth':0, 'EyeR':1, 'EyeL':2,'Tail':3}



from utils import label_map_util

from utils import visualization_utils as vis_util

path_config_file = '/data/home/marrojwala3/fall_2019/DLC/DLC1-JeanM-2019-06-13/config.yaml'

PATH_TO_FROZEN_GRAPH = '/data/home/marrojwala3/objD_test/training_workspace/trained-inference-graphs/output_inference_graph_200k_all_trials_dlc' + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('/data/home/marrojwala3/objD_test/training_workspace/training', 'label_map.pbtxt')



import argparse
parser = argparse.ArgumentParser(description='This script is for inferencing images or videos')

parser.add_argument('-M', action='store', dest='model',default = PATH_TO_FROZEN_GRAPH)
parser.add_argument('-L', action='store', dest = 'parts_label',default = PATH_TO_LABELS)
parser.add_argument('-V' ,action='store', dest = 'videos')
parser.add_argument('-C', action='store', dest = 'config',default = path_config_file)
parser.add_argument('-GPU',action ='store_true',dest= 'gpu_all', default = True )
parser.add_argument('-GPU_to_use', action = 'store',dest = 'gpus_2_use', default = "0")

results = parser.parse_args()

print("Using GPU:" + results.gpus_2_use)



videos = [f for f in glob.glob(results.videos +"/*.h264.mp4", recursive=True)]

print(videos)
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    # with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.9), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(results.model, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

os.environ["CUDA_VISIBLE_DEVICES"] = results.gpus_2_use

category_index = label_map_util.create_category_index_from_labelmap(results.parts_label, use_display_name=True)
with tf.Session(graph=detection_graph) as sess:
    for video in videos:
        print("Processing " + video + " now.\n")
        cap = cv2.VideoCapture(video)
        nframes = int(cap.get(7))
        fps = int(round(cap.get(5)))

        if os.path.isdir(video.split(".")[0]):
            shutil.rmtree(video.split(".")[0])
        os.mkdir(video.split(".")[0])




        IMAGE_SIZE = (12, 8)



        counter = {'frame': [],
                  'count': [],
                  'boxes': []}

        for i in tqdm(range(0,nframes,fps)):
          # image = Image.open(os.path.join(PATH_TO_TEST_IMAGES_DIR, image_path)).convert("RGB")
          # if image.mode != "RGB":
          #       png = image
          #       png.load() # required for png.split()
          #
          #       background = Image.new("RGB", png.size, (255, 255, 255))
          #       background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
          #       image = background

          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          cap.set(1,i)

          ret,frame = cap.read()
          if ret:
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
              image_np =frame[92:962,297:1174,:]
              image_np_expanded = np.expand_dims(image_np, axis=0)
              # Actual detection.


              output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
              # Visualization of the results of a detection.
              vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  output_dict['detection_boxes'],
                  output_dict['detection_classes'],
                  output_dict['detection_scores'],
                  category_index,
                  instance_masks=output_dict.get('detection_masks'),
                  use_normalized_coordinates=True,
                  line_thickness=8)


              # f=0
              # if os.path.isdir(SAVE_PATH +"/../temp"):
              #       shutil.rmtree(SAVE_PATH +"/../temp")
              # os.mkdir(SAVE_PATH +"/../temp")
              f_num=len(output_dict['detection_scores'][output_dict['detection_scores'] > 0.9])
              #
              boxes = []
              for k in output_dict['detection_boxes'][output_dict['detection_scores'] >0.5]:

                  for L in range(4):
                      if i%2 ==0:
                          k[L] = k[L]*image_np.shape[1]
                      else:
                          k[L] = k[L]*image_np.shape[0]
                  boxes.append(k)
              #
              #     plt.imsave(os.path.join(SAVE_PATH +"/../temp",str(f) +".png"),np.array(image.crop((k[1],k[0],k[3],k[2]))))
              #     f +=1
              # if len(os.listdir(SAVE_PATH +"/../temp")) != 0:
              #       deeplabcut.analyze_time_lapse_frames(path_config_file,'/data/home/marrojwala3/fall_2019/obj_d/temp/',save_as_csv=True, frametype='.png')
              #       a = pd.read_hdf('/data/home/marrojwala3/fall_2019/obj_d/temp/tempDeepCut_resnet50_DLC1Jun13shuffle1_1030000.h5')
              #
              #
              #       x_vals = (a.xs('x',axis=1,level=2).T +output_dict['detection_boxes'][output_dict['detection_scores'] >0.5][:,1]*image.size[0]).T
              #       y_vals = (a.xs('y',axis=1,level=2).T +output_dict['detection_boxes'][output_dict['detection_scores'] >0.5][:,0]*image.size[1]).T
              #
              #       x_vals = x_vals[a.xs('likelihood', axis=1,level=2) > 0.5]
              #       y_vals = y_vals[a.xs('likelihood', axis=1,level=2) > 0.5]
              #
              #
              #   for i in col_part.keys():
              #       for j in range(len(x_vals)):
              #           rr,cc =circle(y_vals['DeepCut_resnet50_DLC1Jun13shuffle1_1030000'][i][j],x_vals['DeepCut_resnet50_DLC1Jun13shuffle1_1030000'][i][j], 5,(image_np.shape[0],image_np.shape[1]) )
              #           image_np[rr,cc,:] = colors[col_part[i]]
              # shutil.rmtree(SAVE_PATH +"/../temp")

              # o_im = Image.fromarray(image_np)
              # draw = ImageDraw.Draw(o_im)
              # draw.text((image.size[0]*0.1, image.size[1]*0.95), "Fish Count = " + str(len(output_dict['detection_scores'][output_dict['detection_scores'] > 0.5])), fill = "cyan", font = font)
              #
              if i%9000 == 0:
                  plt.figure(figsize=IMAGE_SIZE)
                  plt.imsave(os.path.join(video.split(".")[0],str(i/1800)+".png"),image_np)
                  plt.close()
              counter['frame'].append(i)
              counter['count'].append(f_num)
              counter['boxes'].append(boxes)





        p = pd.DataFrame.from_dict(counter).sort_values(by='frame')

        p.to_hdf(video.split(".")[0]+".h5", key = 'p',mode='w')
