import json
import os
import argparse
import base64
import csv
import maskrcnn_benchmark.structures.tsv_file_ops as file_ops
from maskrcnn_benchmark.utils.miscellaneous import write_to_yaml_file
import cv2
from tqdm import tqdm

def run(args):
    json_path  = args.json_path
    data_root = args.data_root
    out_dir = args.out_dir
    split = args.split

    with open(json_path) as f:
        objd_image_file = json.load(f)

    train_tsv = []
    train_label_tsv = []
    train_labelmap_tsv = set()
    train_tsv_path = os.path.join(out_dir,split,'train.tsv')
    train_label_path = os.path.join(out_dir,split,'train.label.tsv')
    train_label_map_path = os.path.join(out_dir,split,'train.labelmap.tsv')
    for image in objd_image_file:
        image_path_abs = os.path.join(data_root +  image['image_path'])
        with open(image_path_abs, "rb") as image_file:
            img = cv2.imread(image_path_abs)
            img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img)[1])

        img_objs = []
        for instance in image['obj_instances']:
            img_objs.append({"rect": instance['bbox'], "class": instance['category']})
            train_labelmap_tsv.add(instance['category'])

        train_tsv.append([str(image['image_path']), img_encoded_str.decode('utf-8')])
        row_label = [image['image_path'], json.dumps({"objects":img_objs})]
        train_label_tsv.append(row_label)
    #@TODO replace 'train' with user input
    tsv_split_path = os.path.join(out_dir,split)
    if not os.path.exists(tsv_split_path):
        os.makedirs(tsv_split_path)
    file_ops.tsv_writer(values=train_tsv,tsv_file=train_tsv_path)
    file_ops.tsv_writer(values=train_label_tsv,tsv_file=train_label_path)
    file_ops.generate_hw_file(train_tsv_path)
    file_ops.generate_labelmap_file(train_label_path, os.path.join(tsv_split_path,'train.labelmap.json'))
    file_ops.generate_linelist_file(train_label_path, os.path.join(tsv_split_path,'train.linelist.tsv'))

    yaml_path =  os.path.join(tsv_split_path,'train.yaml')
    yaml_dict = dict()
    yaml_dict['img'] = 'train.tsv'
    yaml_dict['label'] = 'train.label.tsv'
    yaml_dict['hw'] = 'train.hw.tsv'
    yaml_dict['labelmap'] = 'train.labelmap.json'
    yaml_dict['linelist'] = 'train.linelist.tsv'

    write_to_yaml_file(yaml_dict, yaml_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/alfred_splits/')
    parser.add_argument('--json_path', type=str, default='./data/object_detection_jsons/task_0-10.json')
    parser.add_argument('--out_dir', type=str, default='./data/obdet_tsvs/')
    parser.add_argument('--split', type=str, default='task_0-10')
    args = parser.parse_args()
    run(args)