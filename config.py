#### Config
data_path = "/home2/data/"
general = {
    "word_embedings_path": data_path + "images/glove/glove.6B.200d.txt",
    "PL_word_embedings_path": data_path + "images/glovePL/glove_100_3_polish.txt",
    "embedings_dim": 199,
    "PL_embedings_dim": 299,
    "results_directory": "results",
    "coco-caption_path": "./coco-caption",
}

config_mixed_flickr8k_flickr8k = {
    "train": {"dataset_name": "flickr8k", "subset_name": "test"},
    "test": {"dataset_name": "flickr8k", "subset_name": "train"},
    "train_model": True,
    "save_model": True,
    "model_save_dir": "/model_weights/",
    "model_save_path": "/model_weights/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_30k",
}

config_mixed_flickr8k_coco2017 = {
    "train": {"dataset_name": "flickr8k", "subset_name": "test"},
    "test": {"dataset_name": "coco2017", "subset_name": "train"},
    "train_model": True,
    "save_model": True,
    "model_save_dir": "/model_weights/",
    "model_save_path": "/model_weights/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_coco2017",
}

config_flickr8k_30k = {
    "train_images": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test_images": {"dataset_name": "flickr30k", "subset_name": "test"},
    "train_model": True,
    "save_model": True,
    "model_save_dir": "/model_weights/",
    "model_save_path": "/model_weights/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_30k",
}