import os
import glob
import json
import pathlib
import shutil
from object_detection.utils import label_map_util
from sklearn.model_selection import train_test_split




def save_annotations(path_to_json, annotations_dist):
    if not os.path.isfile(path_to_json):
        with open(path_to_json, 'w', encoding='utf-8') as outfile:
            json.dump(annotations_dist, outfile, indent=4)
    else:
        with open(path_to_json) as feedsjson:
            feeds = json.load(feedsjson)
        for item in annotations_dist['images']:
            feeds['images'].append(item)
        with open(path_to_json, 'w', encoding='utf-8') as outfile:
            json.dump(feeds, outfile, indent=4)


def split_dataset_sklearn(path_to_full_dataset, path_to_my_dataset, map_path=None, val_size=0.25):
    annotations_path = os.path.join(path_to_full_dataset, 'annotations.json')
    #update_classes_from_map(map_path, annotations_path)
    with open(annotations_path) as feedsjson:
        json_file = json.load(feedsjson)
    images = json_file['images']
    full_fns= [image['file_name'] for image in images]
    classes = [image['class_text'][0] for image in images]
    train,val = train_test_split(full_fns,stratify=classes,test_size=val_size,random_state=100)
    print("Number of training images: {}".format(len(train)))
    print("Number of validation images: {}".format(len(val)))
    splits = {"train": train, "val": val}
    for key,values in splits.items():
        directory = os.path.join(path_to_my_dataset,key)
        if os.path.exists(directory):
            shutil.rmtree(directory)
        pathlib.Path(directory).mkdir(exist_ok=True)
        for fn in values:
            shutil.copy(os.path.join(path_to_full_dataset,fn),os.path.join(directory,fn))
        annotations = {}
        annotations['images'] = [image for image in images if image['file_name'] in values]
        print("{} annotations: {}".format(key, len(annotations['images'])))
        save_annotations(os.path.join(path_to_my_dataset, key , 'annotations.json'), annotations)






#these two functions used to update annotations and file_names after dataset
#have been created
def update_classes_from_map(map_path, annotations_path):
    labels = label_map_util.get_label_map_dict(map_path)
    with open(annotations_path) as feedsjson:
        json_file = json.load(feedsjson)
    images = json_file['images']
    for image in images:
        image['class'][0] = labels.get(image['class_text'][0])
    json_file['images'] = images
    with open(annotations_path, 'w', encoding='utf-8') as outfile:
        json.dump(json_file, outfile, indent=4)

def rename_class_to_subclass(main_class, subclass, path_to_full_dataset):
    class_files = main_class + '_*.jpg'
    full_fns = glob.glob(path_to_full_dataset + '\\' + class_files)
    for fn in full_fns:
        new_fn = fn.replace(main_class, subclass)
        os.rename(fn, new_fn)
