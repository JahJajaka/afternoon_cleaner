import os
import json
import tensorflow as tf
import cv2
import numpy as np

from object_detection.utils import dataset_util, data_utils

CWD_PATH = os.getcwd()
MY_DATASET = os.path.join(CWD_PATH, 'object_detection', 'datasets', 'my_dataset')
FULL_DATASET = os.path.join(CWD_PATH, 'object_detection', 'datasets', 'my_dataset', 'full_dataset')
MY_CONFIG = os.path.join(CWD_PATH, 'object_detection', 'data', 'afternoon_cleaner_label_map.pbtxt')


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
    data_utils.split_dataset_sklearn(MY_CONFIG, FULL_DATASET, MY_DATASET)
    train_tf_record_path = os.path.join(MY_DATASET, 'train', 'TFRecord')
    train_annotations_path = os.path.join(MY_DATASET, 'train', 'annotations.json')
    train_images_path = os.path.join(MY_DATASET, 'train')
    create_tf_records(train_tf_record_path, train_annotations_path, train_images_path)
    val_tf_record_path = os.path.join(MY_DATASET, 'val', 'TFRecord')
    val_annotations_path = os.path.join(MY_DATASET, 'val', 'annotations.json')
    val_images_path = os.path.join(MY_DATASET, 'val')
    create_tf_records(val_tf_record_path, val_annotations_path, val_images_path)




# In[ ]:
