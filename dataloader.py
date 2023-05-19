import glob
import itertools
import json

import config_datasets
import pickle
from IPython.display import display
from IPython.display import HTML

def split_captions(all_descriptions, train_images, test_images):
    """
        Split captions to train and test sets  and map image id to the set of captions
    Parameters
    ----------
    all_descriptions: dict
        Dictionary with key-image filename, value- list of captions separated by comma.
        Captions are raw, without any text preprocessing, that is specific for NLP tasks.
    train_images: list
        List of image ids that belongs to train set.
    test_images: list
        List of image ids that belongs to test set.

    Returns
    -------
    train_images_mapping: dict
        Dictionary od images that are in train split, with key-image filename,
         value- list of captions separated by comma. Captions are raw, without any text preprocessing,
         that is specific for NLP tasks.
    test_images_mapping: dict
        Dictionary od images that are in test split, with key-image filename,
         value-list of captions separated by comma.Captions are raw, without any text preprocessing,
          that is specific for NLP tasks.
    """
    train_images_mapping = dict()
    test_images_mapping = dict()
    for x in list(all_descriptions.keys()):
        if x in train_images:
            train_images_mapping[x] = all_descriptions[x]
        if x in test_images:
            test_images_mapping[x] = all_descriptions[x]
    return train_images_mapping, test_images_mapping


def get_dataset_configuration(dataset_name):
    """
        Method to get configuration (path to the images, data splits etc) specific for dataset fe. COCO2017, Flickr8k
    Parameters
    ----------
    dataset_name : str
        Name that explicitly identifies name of the dataset from file config_datasets.py
    Returns
    -------
    config: dict
        Dictionary with configuration parameters fe. language of the dataset or path to the directory with images
    """
    if dataset_name == "flickr8k":
        return config_datasets.config_flickr8k
    elif dataset_name == "flickr8k_polish":
        return config_datasets.config_flickr8k_polish
    elif dataset_name == "flickr30k":
        return config_datasets.config_flickr30k
    elif dataset_name == "flickr30k_polish":
        return config_datasets.config_flickr30k_polish
    elif dataset_name == "aide":
        return config_datasets.config_aide
    elif dataset_name == "coco14":
        return config_datasets.config_coco14
    elif dataset_name == "coco17":
        return config_datasets.config_coco17
    else:
        return Exception("Bad name of dataset")


def load_doc(filename):
    """
    Load docuemnt from specigied "filename"
    Parameters
    ----------
    filename: str
        Path to the file to load
    Returns
    -------
    text: str
        Continous text. Need to be read with separators specific for file
    """
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def load_all_captions_flickr(captions_file_path):
    """
    Load all captions from Flickr type dataset
    Parameters
    ----------
    caption_file_path : str
        Path to the file containing all Flickr dataset captions
    Returns
    -------
    all_captions:dict
        Dictionary with key image filename value- list of captions separated by comma.
        Captions are raw, without any text preprocessing specific for NLP tasks.
    """
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


def load_images_coco(configuration):
    """
        Load images from files of COCO dataset
    Parameters
    ----------
    configuration : dict
        Configuration of the datasets used in training from config_datasets.py.
    Returns
    -------
    train_images: dict->{image_filename: global path to the image}
        train split of images
    test_images: dict->{image_filename: global path to the image}
        test split of images
    """
    file_with_images_def = configuration["images_names_file_path"]
    # directory will all images from COCO dataset
    images_folder = configuration["images_dir"]
    info = json.load(open(file_with_images_def))
    train_images_mapping = dict()
    test_images_mapping = dict()
    # iterate over all images from COCO dataset listed in file configuration["images_names_file_path"]
    for ix in range(len(info['images'])):

        img = info['images'][ix]
        # remove from image name .jpg sufix
        image_filename = img['file_path'].rsplit(".", 1)[0]
        # add global path to the image name
        file_path = images_folder + "/" + img['file_path']

        if image_filename.find("/") != -1:
            image_filename = img['file_path'].rsplit("/", 1)[1].rsplit(".", 1)[0]
        # assignment of images from specific split
        # in our case we use train and test dataset
        # all images assigned to restval, train by Karpathy split are assigned to the train split
        # images from val and test are assigned to the test split
        if img['split'] == 'train':
            train_images_mapping[image_filename] = file_path
        # elif img['split'] == 'val':
        #     test_images_mapping[image_filename] = file_path
        elif img['split'] == 'test':
            test_images_mapping[image_filename] = file_path
        elif img['split'] == 'restval':
            train_images_mapping[image_filename] = file_path

    return train_images_mapping, test_images_mapping


def load_all_captions_coco(caption_file_path):
    """
    Load all captions from cocodataset
    Parameters
    ----------
    caption_file_path : str
        Path to the file containing COCO dataset captions
    Returns
    -------
    all_captions:dict->{"232332": []
        Dictionary with key image filename value- list of captions separated by comma.
        Captions are raw, without any text preprocessing specific for NLP tasks.
    """
    # open file with coco captions
    imgs = json.load(open(caption_file_path, 'r'))
    imgs = imgs['images']
    all_captions = dict()
    for img in imgs:
        # remove ".jpg" sufix
        image_filename = img['filename'].rsplit(".", 1)[0]
        # create dictionary
        # in the file key "raw" represents captions without any processing
        if image_filename not in all_captions:
            all_captions[image_filename] = list()
        for sent in img['sentences']:
            all_captions[image_filename].append(sent['raw'])

    return all_captions


def load_images_flickr(images_dir, train_images_file_path, test_images_file_path):
    """Method to map images ids to pictures

    Parameters
    ----------
    images_dir: str
        Path to the directory with all images from  Flickr type dataset
    train_images_file_path
        Path to the file with image names of images from train split
    test_images_file_path
        Path to the file with image names of images from test split
    Returns
    -------
    train_images_mapping: dict->{image_filename: global path to the image}
        train split of images
    test_images_mapping: dict->{image_filename: global path to the image}
        test split of images

    """
    train_images = set(open(train_images_file_path, 'r').read().strip().split('\n'))
    train_images_mapping = dict()
    test_images = set(open(test_images_file_path, 'r').read().strip().split('\n'))
    test_images_mapping = dict()
    # add global paths to the all images in images_dir directory
    all_images = glob.glob(images_dir + '*.jpg')
    for i in all_images:  # img is list of full path names of all images
        image_name = i.split("/")[-1]
        image_id = image_name.split(".")[0]
        if image_name in train_images:  # Check if the image belongs to train set
            train_images_mapping[image_id] = i  # Add it to the dict of train images
        if image_name in test_images:  # Check if the image belongs to test set
            test_images_mapping[image_id] = i  # Add it to the dict of test images
    return train_images_mapping, test_images_mapping


def load_dataset(configuration):
    """General method to load train and test set into framework.
        Framework accepts different types of datasets in test and train split, fe. train - COCO2017, test - Flickr8k.
        Following this assumption, datasets for train and test sets are loaded separately.
        All data for all datasets are loaded (all captions, all images for training and testing) and separated
        accordingly to the splits defined in configurations files. Therefore user can freely mix con
         mix data configurations for training and testing.
    Parameters
    ----------
    configuration : dict
        Configuration of the datasets used in training.

    Returns
    -------
    train: dict->{
                "train_images_mapping_original": dict,
                "test_images_mapping_original": dict,
                "all_captions": dict,
                "train_captions_mapping_original": dict,
                "test_captions_mapping_original": dict
            }
            Dictionary with data assigned to the training split.
    test: dict->{
                "train_images_mapping_original": dict,
                "test_images_mapping_original": dict,
                "all_captions": dict,
                "train_captions_mapping_original": dict,
                "test_captions_mapping_original": dict
            }
            Dictionary with data assigned to the testing split.
    """

    def get_data_for_split(split_name):
        # Load dataset configuration, by the name of the dataset assigned for training/testing
        dataset_configuration = get_dataset_configuration(configuration[split_name]["dataset_name"])
        # Therefore Flickr and COCO have different file and data structures, to show captions and split of data
        # different methods for loading captions and images are used.
        # Datasets Flickr30k, COCO2017, COCO2014 have the same strucutre of files with captions and split informations.
        if dataset_configuration["data_name"] in ["flickr30k", "coco17", "coco14"]:
            # Load train images and test images and assign them to specific splits
            print("Loading images splits")
            train_images_mapping_original, test_images_mapping_original = load_images_coco(dataset_configuration)
            print("Images splits loaded")
            print("Number of train images: ", len(train_images_mapping_original))
            print("Number of test images: ", len(test_images_mapping_original))
            # Load all captions from dataset, that is COCO type
            print("Loading all captions")
            all_captions = load_all_captions_coco(dataset_configuration["captions_file_path"])
            print("All captions loaded")
            print("Nuber of all captions: ", len(all_captions))
        # Datasets Flickr30k, Flickr8k_polish, AIDe, Flickr8k  have the same strucutre of files with captions and split informations.
        if dataset_configuration["data_name"] in ["flickr30k_polish", "flickr8k_polish", "aide", "flickr8k"]:
            # Load train images and test images and assign them to specific splits
            print("Loading images splits")
            train_images_mapping_original, test_images_mapping_original = load_images_flickr(
                dataset_configuration["images_dir"], dataset_configuration[
                    "train_images_names_file_path"],
                dataset_configuration[
                    "test_images_names_file_path"])
            print("Images splits loaded")
            print("Number of train images: ", len(train_images_mapping_original))
            print("Number of test images: ", len(test_images_mapping_original))
            # Load all captions from dataset, that is Flickr8k type
            print("Loading all captions")
            all_captions = load_all_captions_flickr(dataset_configuration["captions_file_path"])
            print("All captions loaded")
            print("Nuber of all captions: ", len(all_captions))
        # Assign captions to specific splits
        print("Loading captions splits")
        train_captions_mapping_original, test_captions_mapping_original = split_captions(all_captions,
                                                                                         list(
                                                                                             train_images_mapping_original.keys()),
                                                                                         list(
                                                                                             test_images_mapping_original.keys()))
        c_train_captions_mapping_original = list(itertools.chain.from_iterable(train_captions_mapping_original))
        print(c_train_captions_mapping_original)
        c_test_captions_mapping_original = list(itertools.chain.from_iterable(test_captions_mapping_original))
        print(c_test_captions_mapping_original)
        with open("train_captions.pkl", 'w+b') as encoded_pickle:
            pickle.dump(c_train_captions_mapping_original, encoded_pickle)
        with open("test_captions.pkl", 'w+b') as encoded_pickle:
            pickle.dump(c_test_captions_mapping_original, encoded_pickle)
        print("Captions splits loaded")
        print("Number of train captions: ", len(train_captions_mapping_original))
        print("Number of test test: ", len(test_captions_mapping_original))
        return {
            "train": {
                "train_images_mapping_original": train_images_mapping_original,
                "train_captions_mapping_original": train_captions_mapping_original
            },
            "test": {
                "test_images_mapping_original": test_images_mapping_original,
                "test_captions_mapping_original": test_captions_mapping_original
            },
            "all_captions": all_captions,
            "language": dataset_configuration['language']
        }

    print("Loading train dataset")
    train = get_data_for_split("train")
    print("Loading test dataset")
    test = get_data_for_split("test")
    language = train['language']
    return train, test, language


class DataLoader:
    def __init__(self, configuration):
        print("Loading dataset")
        self.train, self.test, self.language = load_dataset(configuration)
        self.configuration = configuration
