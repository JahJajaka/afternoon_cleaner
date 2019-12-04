import os
import json
import tensorflow as tf
import cv2
import numpy as np
import glob
from tqdm import tqdm_notebook

from object_detection.utils import dataset_util, data_utils, label_map_util
from object_detection.utils.app_utils import load_yaml, draw_boxes_and_labels

CWD_PATH = os.getcwd()
config_folder =os.path.join(CWD_PATH, 'config.yaml')
cfg = load_yaml(config_folder)

MY_DATASET = os.path.join(CWD_PATH, 'object_detection', 'datasets', 'my_dataset')
FULL_DATASET= os.path.join(CWD_PATH, os.path.abspath(cfg['MY_DATASET']))

MODEL_NAME = cfg['MODEL_NAME']
MODELS_DIR = os.path.join(CWD_PATH, 'object_detection', 'datasets')
PATH_TO_LABELS = os.path.join(CWD_PATH, os.path.abspath(cfg['PATH_TO_LABELS']))
PATH_TO_MY_LABELS = os.path.join(CWD_PATH, os.path.abspath(cfg['PATH_TO_MY_LABELS']))
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', 'datasets', MODEL_NAME, 'frozen_inference_graph.pb')
path_to_annotations = os.path.join(FULL_DATASET, 'annotations.json')
IMAGE_PATHS = glob.glob("{}/*.jpg".format(FULL_DATASET))
NUM_CLASSES =  cfg['NUM_CLASSES']




# Patch the location of gfile
tf.gfile = tf.io.gfile
my_labels = label_map_util.get_label_map_dict(PATH_TO_MY_LABELS)
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)



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
        min_score_thresh=0.5
    )

    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)



def create_annotations(images_path, my_labels, path_to_annotations, height, width):
    annotations = {}
    annotations['images'] = []
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)
    for image_path in tqdm_notebook(images_path):
        file_name = image_path.split('\\')[-1]
        main_class = file_name.split('_')[0]
        subclass = file_name.split('_')[1]
        image_np = cv2.imread(image_path)
        data = detect_objects(image_np, sess, detection_graph)
        rec_points = data['rect_points']
        class_names = data['class_names']
        for point, name in zip(rec_points, class_names):
            name_only = name[0].split(':')[0]
            if name_only == main_class:
                annotations['images'].append({
                        'height': height,
                        'width': width,
                        'file_name': file_name,
                        'xmin': [point['xmin']],
                        'xmax': [point['xmax']],
                        'ymin': [point['ymin']],
                        'ymax': [point['ymax']],
                        'image_format': 'jpg',
                        'main_class': [name_only],
                        'class_text': [subclass],
                        'class': [my_labels.get(subclass)]
                })
    print("{} objects have been annotated".format(len(annotations['images'])))
    data_utils.save_annotations(path_to_annotations, annotations)





def create_tf_example(example, path_to_images):
    img = cv2.imread(os.path.join(path_to_images, example['file_name']))
    img_encoded = cv2.imencode('.jpg', img)[1]
    img_bytes = img_encoded.tobytes()
    height = example['height'] # Image height
    width = example['width'] # Image width
    filename = example['file_name'].encode('utf-8') # Filename of the image. Empty if image is not from file
    encoded_image_data = img_bytes # Encoded image bytes
    image_format = example['image_format'].encode('utf-8') # b'jpeg' or b'png'

    xmins = example['xmin'] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = example['xmax'] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = example['ymin'] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = example['ymax'] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = np.array([class_text.encode('utf-8') for class_text in example['class_text']]) # List of string class name of bounding box (1 per box)
    classes = example['class'] # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example



def create_tf_records(path_to_tf_records, path_to_annotations, path_to_images):
    writer = tf.io.TFRecordWriter(path_to_tf_records)
    with open(path_to_annotations) as feedsjson:
        json_file = json.load(feedsjson)
    examples = json_file['images']
    for example in examples:
        tf_example = create_tf_example(example, path_to_images)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecords: {}'.format(path_to_tf_records))


if __name__ == '__main__':
    #create annotations.json file from collected images
    create_annotations(IMAGE_PATHS, my_labels, path_to_annotations, cfg['ARGS']['HEIGHT'], cfg['ARGS']['WIDTH'])  
    #split data to train and val datasets
    data_utils.split_dataset_sklearn(PATH_TO_MY_LABELS, FULL_DATASET, MY_DATASET)
    #create train_tf_record
    train_tf_record_path = os.path.join(MY_DATASET, 'train', 'TFRecord')
    train_annotations_path = os.path.join(MY_DATASET, 'train', 'annotations.json')
    train_images_path = os.path.join(MY_DATASET, 'train')
    create_tf_records(train_tf_record_path, train_annotations_path, train_images_path)
    #create val_tf_record
    val_tf_record_path = os.path.join(MY_DATASET, 'val', 'TFRecord')
    val_annotations_path = os.path.join(MY_DATASET, 'val', 'annotations.json')
    val_images_path = os.path.join(MY_DATASET, 'val')
    create_tf_records(val_tf_record_path, val_annotations_path, val_images_path)




# In[ ]:
