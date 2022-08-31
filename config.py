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
config_mixed_flickr8k_polish_flickr8k_polish = {
    "train": {"dataset_name": "flickr8k_polish", "subset_name": "test"},
    "test": {"dataset_name": "flickr8k_polish", "subset_name": "train"},
    "train_model": True,
    "save_model": True,
    "model_save_dir": "/model_weights/",
    "model_save_path": "/model_weights/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_polish_flickr8k_polish",
}
config_mixed_aide_aide = {
    "train": {"dataset_name": "aide", "subset_name": "test"},
    "test": {"dataset_name": "aide", "subset_name": "train"},
    "train_model": True,
    "save_model": True,
    "model_save_dir": "/model_weights/",
    "model_save_path": "/model_weights/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_aide_aide",
}
config_mixed_coco2017_coco2017 = {
    "train": {"dataset_name": "coco17", "subset_name": "test"},
    "test": {"dataset_name": "coco17", "subset_name": "train"},
    "train_model": True,
    "save_model": True,
    "model_save_dir": "/model_weights/",
    "model_save_path": "/model_weights/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco2017_coco2017",
}
config_mixed_coco2014_coco2014 = {
    "train": {"dataset_name": "coco14", "subset_name": "test"},
    "test": {"dataset_name": "coco14", "subset_name": "train"},
    "train_model": True,
    "save_model": True,
    "model_save_dir": "/model_weights/",
    "model_save_path": "/model_weights/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco2014_coco2014",
}
config_mixed_coco2017_flickr8k = {
    "train": {"dataset_name": "coco17", "subset_name": "test"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "train_model": True,
    "save_model": True,
    "model_save_dir": "/model_weights/",
    "model_save_path": "/model_weights/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco2017_flickr8k",
}