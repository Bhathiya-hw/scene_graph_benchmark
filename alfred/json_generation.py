import json
import os
import argparse
import cv2

def run(args):
    data_root = args.data_root
    split = args.split
    input_folder = os.path.join(data_root, split)
    all_jsons= [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_folder) for f in filenames
                if os.path.splitext(f)[1] == '.json' and  'metadata' not in os.path.splitext(f)[0]]

    output_file = args.output_file
    metadata_list = []
    for json_path in all_jsons:

        with open(json_path) as f:
            image_meta = json.load(f)

        detection_data = dict()

        image_path = json_path.replace(data_root, '').replace('.json', '.jpg')
        detection_data['image_path'] = image_path
        detection_data['height'] = image_meta['height']
        detection_data['width'] = image_meta['width']
        detection_data['split'] = split


        anns = image_meta['annotations']
        instances = []
        for inst  in anns:
            instance = dict()
            instance['bbox'] = inst['bbox']
            instance['bbox_mode'] = inst['bbox_mode']
            instance['category'] = inst['category']
            instances.append(instance)

        detection_data['obj_instances'] = instances
        metadata_list.append(detection_data)

    with open(output_file,'w+') as out:
        json.dump(metadata_list,out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--split', type=str, )
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()
    run(args)