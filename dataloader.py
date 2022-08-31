import glob
import json

import config_datasets


def load_captions(all_descriptions, train_images_file_path, test_images_file_path):
    train_images = set(open(train_images_file_path, 'r').read().strip().split('\n'))
    train_images_mapping = dict()
    test_images = set(open(test_images_file_path, 'r').read().strip().split('\n'))
    test_images_mapping = dict()
    for x in list(all_descriptions.keys()):
        if x in train_images:
            train_images_mapping[x] = all_descriptions[x]
        if x in test_images:
            test_images_mapping[x] = all_descriptions[x]
    return train_images_mapping, test_images_mapping


def get_dataset_configuration(dataset_name):
    if dataset_name is "flickr8k":
        return config_datasets.config_flickr8k
    elif dataset_name is "flickr8k_polish":
        return config_datasets.config_flickr8k_polish
    elif dataset_name is "flickr30k":
        return config_datasets.config_flickr30k
    elif dataset_name is "flickr30k_polish":
        return config_datasets.config_flickr30k_polish
    elif dataset_name is "aide":
        return config_datasets.config_aide
    elif dataset_name is "coco14":
        return config_datasets.config_coco14
    elif dataset_name is "coco17":
        return config_datasets.config_coco17


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def load_descriptions_flickr(captions_file_path):
    doc = load_doc(captions_file_path)
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # extract filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
        # store description
        mapping[image_id].append(image_desc)
    return mapping


def load_coco_data(configuration):
    input_json = configuration["images_file_path"]
    images_folder = configuration["images_dir"]
    info = json.load(open(input_json))
    train_images = dict()
    test_images = dict()
    for ix in range(len(info['images'])):
        img = info['images'][ix]
        image_filename = img['file_path']
        file_path = images_folder + "/" + img['file_path']

        if image_filename.find("/") != -1:
            image_filename = img['file_path'].rsplit("/", 1)[1]

        if img['split'] == 'train':
            train_images[image_filename] = file_path
        elif img['split'] == 'val':
            test_images[image_filename] = file_path
        elif img['split'] == 'test':
            test_images[image_filename] = file_path
        elif img['split'] == 'restval':
            train_images[image_filename] = file_path

    return train_images, test_images


def get_all_train_captions(train_descriptions):
    # Create a list of all the training captions
    all_train_captions = []
    for key, val in train_descriptions.items():
        for cap in val:
            all_train_captions.append(cap)
    return all_train_captions


def load_captions_coco(filename, train_images_names_list, test_images_names_list):
    imgs = json.load(open(filename, 'r'))
    imgs = imgs['images']
    all_captions = dict()
    for img in imgs:
        image_filename = img['filename']
        # create list
        if image_filename not in all_captions:
            all_captions[image_filename.rsplit(".", 1)[0]] = list()
        for sent in img['sentences']:
            all_captions[image_filename.rsplit(".", 1)[0]].append(sent['raw'])

    return all_captions


def images_with_path(images_dir, train_images_file_path, test_images_file_path):
    train_images = set(open(train_images_file_path, 'r').read().strip().split('\n'))
    train_images_mapping = dict()
    test_images = set(open(test_images_file_path, 'r').read().strip().split('\n'))
    test_images_mapping = dict()
    all_images = glob.glob(images_dir + '*.jpg')
    for i in all_images:  # img is list of full path names of all images
        image_name = i.split("\n")[-1]
        image_id = image_name.split(".")[0]
        if image_name in train_images:  # Check if the image belongs to test/training set
            train_images_mapping[image_id] = i  # Add it to the list of test/train images
        if image_name in test_images:  # Check if the image belongs to test/training set
            test_images_mapping[image_id] = i
    return train_images_mapping, test_images_mapping


def load_dataset(configuration):
    def get_data_for_split(configuration, split_name):
        train_dataset_configuration = get_dataset_configuration(configuration[split_name]["dataset_name"])
        if train_dataset_configuration["data_name"] in ["flickr30k", "coco17", "coco14"]:
            train_images_mapping_original, test_images_mapping_original = load_coco_data(train_dataset_configuration)
            all_captions = load_captions_coco(train_dataset_configuration["captions_file_path"],
                                              list(train_images_mapping_original.keys()),
                                              list(test_images_mapping_original.keys()))
            train_captions_mapping_original, test_captions_mapping_original = load_captions(all_captions,
                                                                                            train_dataset_configuration[
                                                                                                "train_images_names_file_path"],
                                                                                            train_dataset_configuration[
                                                                                                "test_images_names_file_path"])
        if train_dataset_configuration["data_name"] in ["flickr30k_polish", "flickr8k_polish", "aide", "flickr8k"]:
            train_images_mapping_original, test_images_mapping_original = images_with_path(
                train_dataset_configuration["images_dir"], train_dataset_configuration[
                    "train_images_names_file_path"],
                train_dataset_configuration[
                    "test_images_names_file_path"])
            all_captions = load_descriptions_flickr(train_dataset_configuration["captions_file_path"])
            train_captions_mapping_original, test_captions_mapping_original = load_captions(all_captions,
                                                                                            train_dataset_configuration[
                                                                                                "train_images_names_file_path"],
                                                                                            train_dataset_configuration[
                                                                                                "test_images_names_file_path"])
        return {"train_images_mapping_original": train_images_mapping_original,
                "test_images_mapping_original": test_images_mapping_original,
                "all_captions": all_captions,
                "train_captions_mapping_original": train_captions_mapping_original,
                "test_captions_mapping_original": test_captions_mapping_original}

    train = get_data_for_split(configuration, "train")
    test = get_data_for_split(configuration, "test")
    train = {}
    return train, test


class DataLoader:
    def __init__(self, configuration):
        self.train, self.test = load_dataset(configuration )
        # self.train_img, self.train_images = self.images_with_path(self.config["train_images_path"])
        # full_train_dataset
        # full_test_dataset
