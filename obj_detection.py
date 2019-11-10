#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import cv2
import time
import argparse
import easydict
import numpy  as np
import shutil
import subprocess as sp
import json
import tarfile
import six.moves.urllib as urllib
import tensorflow as tf
import requests

from queue import Queue
from threading import Thread
from object_detection.utils.app_utils import load_yaml, FPS, HLSVideoStream, WebcamVideoStream, draw_boxes_and_labels
from object_detection.utils import label_map_util


# In[2]:
CWD_PATH = os.getcwd()
config_folder =os.path.join(CWD_PATH, 'config.yaml')
cfg = load_yaml(config_folder)
# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME =cfg['MODEL_NAME']
MODEL_FILE = MODEL_NAME + '.tar.gz'
#data = urllib.parse.urlencode(MODEL_FILE).encode("utf-8")
#PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', 'datasets', MODEL_NAME, 'tflite_graph.pb')
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', 'datasets', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES =  cfg['NUM_CLASSES']

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#Download MODEL
if not os.path.exists(PATH_TO_CKPT):
    DOWNLOAD_BASE = cfg['DOWNLOAD_BASE']
    DOWONLOAD_DIRECTORY = os.path.join(CWD_PATH, 'object_detection')
    model_dir = tf.keras.utils.get_file(fname=MODEL_NAME,origin=DOWNLOAD_BASE + MODEL_FILE,untar=True,cache_dir=DOWONLOAD_DIRECTORY)
# In[3]:



def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    #all_nodes = [n.name for n in detection_graph.as_graph_def().node]
    #print(all_nodes)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')


    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=category_index,
        instance_masks=None,
        min_score_thresh=cfg['MIN_SCORE_THRESH']
    )

    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)


# In[4]:




def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))
    fps.stop()
    sess.close()





def recognition(robot_q):
    parser = argparse.ArgumentParser()
    #parser.add_argument('-strin', '--stream-input', dest="stream_in", action='store', type=str, default=None)
    #parser.add_argument('-src', '--source', dest='video_source', type=str,
    #                    default='http://admin:@192.168.0.9:80/media/?action=stream', help='Device index of the camera.')
    #parser.add_argument('-wd', '--width', dest='width', type=int,
    #                    default=640, help='Width of the frames in the video stream.')
    #parser.add_argument('-ht', '--height', dest='height', type=int,
    #                    default=480, help='Height of the frames in the video stream.')
    #parser.add_argument('-strout','--stream-output', dest="stream_out", help='The URL to send the livestreamed object detection to.')
    #parser.add_argument('cnf', '--config', dest="config", type=str, help='train config path')
    #args = parser.parse_args()

    args = easydict.EasyDict({
            "stream_in": cfg['ARGS']['STREAM_IN'],
            "video_source": cfg['ARGS']['VIDEO_SOURCE'],
            "width": cfg['ARGS']['WIDTH'],
            "height": cfg['ARGS']['HEIGHT'],
            "stream_out": cfg['ARGS']['STREAM_OUT']
             })






    input_q = Queue(1)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    if (args.stream_in):
        print('Reading from hls stream.')
        video_capture = HLSVideoStream(src=args.stream_in).start()
    else:
        print('Reading from webcam.')
        video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 10.0, (640,480))
    fps = FPS().start()

    while True:
        frame = video_capture.read()
        frame = cv2.resize(frame, (args.width,args.height))
        input_q.put(frame)

        t = time.time()

        if output_q.empty():
            pass  # fill up queue
        else:

            font = cv2.FONT_HERSHEY_SIMPLEX
            data = output_q.get()
            rec_points = data['rect_points']
            class_names = data['class_names']
            class_colors = data['class_colors']
            for point, name, color in zip(rec_points, class_names, class_colors):
                name_only = name[0].split(':')[0]
                if name_only in cfg['DETECTED']:
                    cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
                                  (int(point['xmax'] * args.width), int(point['ymax'] * args.height)), color, 3)
                    cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
                                  (int(point['xmin'] * args.width) + len(name[0]) * 6,
                                   int(point['ymin'] * args.height) - 10), color, -1, cv2.LINE_AA)
                    cv2.putText(frame, name[0], (int(point['xmin'] * args.width), int(point['ymin'] * args.height)), font,
                              0.3, (0, 0, 0), 1)
                    robot_q.put(name_only)
                    if robot_q.qsize()>2:
                        with robot_q.mutex:
                            robot_q.queue.clear()
            if args.stream_out:
                print('Streaming elsewhere!')
            else:
                out.write(frame)
                cv2.imshow('Video', frame)

        fps.update()


        if cv2.waitKey(1) & 0xFF == ord('q'):
            robot_q.put("exit")
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    video_capture.stop()
    out.release()
    cv2.destroyAllWindows()
