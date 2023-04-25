#### Config
data_path = "/home2/data/"
general = {
    "results_directory": "./results",
    "coco-caption_path": "./coco-caption",
    "pl_spacy_model": data_path + 'images/pl_spacy_model',
    "START": 'START',
    "STOP": 'STOP',
    "word_count_threshold": 10
}
glove = {
    "eng": {
        "word_embedings_path": data_path + "images/glove/glove.6B.200d.txt",
        "embedings_dim": 199
    }
}
bert = {
    "eng": {
        "word_embedings_path": data_path + "images/glove/glove.6B.200d.txt",
        "embedings_dim": 768
    }
}
fastText = {
    "eng": {
        "word_embedings_path": data_path + "text_models/fastText/wiki-news-300d-1M-subword.vec",
        "embedings_dim": 300
    }
}
word2Vec = {
    "eng": {
        "word_embedings_path": data_path + "text_models/word2Vec/GoogleNews-vectors-negative300.bin",
        "embedings_dim": 300
    }
}
# sprawdzić czy w kazdym konfigu poprawnie jest wpisana data_name





config_mixed_coco14_flickr8k_vgg16_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr8k_vgg16_glove",
    "images_processor": "vgg16",
    "text_processor": "glove"
}
config_mixed_coco14_flickr8k_vgg16_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr8k_vgg16_fastText",
    "images_processor": "vgg16",
    "text_processor": "fastText"
}




config_mixed_coco14_flickr8k_resnet152V2_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr8k_resnet152V2_glove",
    "images_processor": "resnet152V2",
    "text_processor": "glove"
}
config_mixed_coco14_flickr8k_resnet152V2_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr8k_resnet152V2_fastText",
    "images_processor": "resnet152V2",
    "text_processor": "fastText"
}




config_mixed_coco14_flickr8k_Xception_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr8k_Xception_glove",
    "images_processor": "Xception",
    "text_processor": "glove"
}
config_mixed_coco14_flickr8k_Xception_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr8k_Xception_fastText",
    "images_processor": "Xception",
    "text_processor": "fastText"
}



config_mixed_coco14_flickr8k_inception_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr8k_inception_glove",
    "images_processor": "inception",
    "text_processor": "glove"
}
config_mixed_coco14_flickr8k_inception_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr8k_inception_fastText",
    "images_processor": "inception",
    "text_processor": "fastText"
}









config_mixed_coco14_flickr8k_vgg19_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr8k_vgg19_glove",
    "images_processor": "vgg19",
    "text_processor": "glove"
}
config_mixed_coco14_flickr8k_vgg19_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr8k_vgg19_fastText",
    "images_processor": "vgg19",
    "text_processor": "fastText"
}








config_mixed_coco14_flickr8k_resnet50_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr8k_resnet50_glove",
    "images_processor": "resnet50",
    "text_processor": "glove"
}
config_mixed_coco14_flickr8k_resnet50_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr8k_resnet50_fastText",
    "images_processor": "resnet50",
    "text_processor": "fastText"
}









config_mixed_coco14_coco14_vgg16_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_vgg16_glove",
    "images_processor": "vgg16",
    "text_processor": "glove"
}
config_mixed_coco14_coco14_vgg16_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_vgg16_fastText",
    "images_processor": "vgg16",
    "text_processor": "fastText"
}






config_mixed_coco14_coco14_resnet152V2_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_resnet152V2_glove",
    "images_processor": "resnet152V2",
    "text_processor": "glove"
}
config_mixed_coco14_coco14_resnet152V2_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_resnet152V2_fastText",
    "images_processor": "resnet152V2",
    "text_processor": "fastText"
}





config_mixed_coco14_coco14_Xception_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_Xception_glove",
    "images_processor": "Xception",
    "text_processor": "glove"
}
config_mixed_coco14_coco14_Xception_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_Xception_fastText",
    "images_processor": "Xception",
    "text_processor": "fastText"
}








config_mixed_coco14_coco14_inception_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_inception_glove",
    "images_processor": "inception",
    "text_processor": "glove"
}
config_mixed_coco14_coco14_inception_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_inception_fastText",
    "images_processor": "inception",
    "text_processor": "fastText"
}


config_mixed_coco14_coco14_vgg19_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_vgg19_glove",
    "images_processor": "vgg19",
    "text_processor": "glove"
}
config_mixed_coco14_coco14_vgg19_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_vgg19_fastText",
    "images_processor": "vgg19",
    "text_processor": "fastText"
}


config_mixed_coco14_coco14_resnet50_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_resnet50_glove",
    "images_processor": "resnet50",
    "text_processor": "glove"
}
config_mixed_coco14_coco14_resnet50_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_resnet50_fastText",
    "images_processor": "resnet50",
    "text_processor": "fastText"
}

config_mixed_coco14_coco14_denseNet121_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_denseNet121_glove",
    "images_processor": "denseNet121",
    "text_processor": "glove"
}
config_mixed_coco14_coco14_denseNet121_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_denseNet121_fastText",
    "images_processor": "denseNet121",
    "text_processor": "fastText"
}

config_mixed_coco14_coco14_denseNet201_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_denseNet201_glove",
    "images_processor": "denseNet201",
    "text_processor": "glove"
}
config_mixed_coco14_coco14_denseNet201_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_denseNet201_fastText",
    "images_processor": "denseNet201",
    "text_processor": "fastText"
}


config_mixed_coco14_coco14_mobileNet_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_mobileNet_glove",
    "images_processor": "mobileNet",
    "text_processor": "glove"
}
config_mixed_coco14_coco14_mobileNet_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_mobileNet_fastText",
    "images_processor": "mobileNet",
    "text_processor": "fastText"
}

config_mixed_coco14_coco14_mobileNetV2_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_mobileNetV2_glove",
    "images_processor": "mobileNetV2",
    "text_processor": "glove"
}
config_mixed_coco14_coco14_mobileNetV2_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "coco14", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_coco14_mobileNetV2_fastText",
    "images_processor": "mobileNetV2",
    "text_processor": "fastText"
}



config_mixed_flickr8k_flickr8k_vgg16_glove = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_vgg16_glove",
    "images_processor": "vgg16",
    "text_processor": "glove"
}
config_mixed_flickr8k_flickr8k_vgg16_fastText = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_vgg16_fastText",
    "images_processor": "vgg16",
    "text_processor": "fastText"
}



config_mixed_flickr8k_flickr8k_resnet152V2_glove = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_resnet152V2_glove",
    "images_processor": "resnet152V2",
    "text_processor": "glove"
}
config_mixed_flickr8k_flickr8k_resnet152V2_fastText = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_resnet152V2_fastText",
    "images_processor": "resnet152V2",
    "text_processor": "fastText"
}










config_mixed_flickr8k_flickr8k_Xception_glove = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_Xception_glove",
    "images_processor": "Xception",
    "text_processor": "glove"
}
config_mixed_flickr8k_flickr8k_Xception_fastText = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_Xception_fastText",
    "images_processor": "Xception",
    "text_processor": "fastText"
}










config_mixed_flickr8k_flickr8k_inception_glove = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_inception_glove",
    "images_processor": "inception",
    "text_processor": "glove"
}
config_mixed_flickr8k_flickr8k_inception_fastText = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_inception_fastText",
    "images_processor": "inception",
    "text_processor": "fastText"
}






config_mixed_flickr8k_flickr8k_vgg19_glove = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_vgg19_glove",
    "images_processor": "vgg19",
    "text_processor": "glove"
}
config_mixed_flickr8k_flickr8k_vgg19_fastText = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_vgg19_fastText",
    "images_processor": "vgg19",
    "text_processor": "fastText"
}





config_mixed_flickr8k_flickr8k_resnet50_glove = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_resnet50_glove",
    "images_processor": "resnet50",
    "text_processor": "glove"
}
config_mixed_flickr8k_flickr8k_resnet50_fastText = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_resnet50_fastText",
    "images_processor": "resnet50",
    "text_processor": "fastText"
}
config_mixed_flickr8k_flickr8k_denseNet121_glove = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_denseNet121_glove",
    "images_processor": "denseNet121",
    "text_processor": "glove"
}
config_mixed_flickr8k_flickr8k_denseNet121_fastText = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_denseNet121_fastText",
    "images_processor": "denseNet121",
    "text_processor": "fastText"
}
config_mixed_flickr8k_flickr8k_denseNet201_glove = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_denseNet201_glove",
    "images_processor": "denseNet201",
    "text_processor": "glove"
}
config_mixed_flickr8k_flickr8k_denseNet201_fastText = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_denseNet201_fastText",
    "images_processor": "denseNet201",
    "text_processor": "fastText"
}

config_mixed_flickr8k_flickr8k_mobileNet_glove = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_mobileNet_glove",
    "images_processor": "mobileNet",
    "text_processor": "glove"
}
config_mixed_flickr8k_flickr8k_mobileNet_fastText = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_mobileNet_fastText",
    "images_processor": "mobileNet",
    "text_processor": "fastText"
}
config_mixed_flickr8k_flickr8k_mobileNetV2_glove = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_mobileNetV2_glove",
    "images_processor": "mobileNetV2",
    "text_processor": "glove"
}
config_mixed_flickr8k_flickr8k_mobileNetV2_fastText = {
    "train": {"dataset_name": "flickr8k", "subset_name": "train"},
    "test": {"dataset_name": "flickr8k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_flickr8k_flickr8k_mobileNetV2_fastText",
    "images_processor": "mobileNetV2",
    "text_processor": "fastText"
}









config_mixed_coco14_flickr30k_vgg16_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr30k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr30k_vgg16_glove",
    "images_processor": "vgg16",
    "text_processor": "glove"
}
config_mixed_coco14_flickr30k_vgg16_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr30k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr30k_vgg16_fastText",
    "images_processor": "vgg16",
    "text_processor": "fastText"
}


config_mixed_coco14_flickr30k_resnet152V2_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr30k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr30k_resnet152V2_glove",
    "images_processor": "resnet152V2",
    "text_processor": "glove"
}
config_mixed_coco14_flickr30k_resnet152V2_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr30k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr30k_resnet152V2_fastText",
    "images_processor": "resnet152V2",
    "text_processor": "fastText"
}




config_mixed_coco14_flickr30k_vgg19_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr30k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr30k_vgg19_glove",
    "images_processor": "vgg19",
    "text_processor": "glove"
}
config_mixed_coco14_flickr30k_vgg19_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr30k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr30k_vgg19_fastText",
    "images_processor": "vgg19",
    "text_processor": "fastText"
}






config_mixed_coco14_flickr30k_Xception_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr30k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr30k_Xception_glove",
    "images_processor": "Xception",
    "text_processor": "glove"
}
config_mixed_coco14_flickr30k_Xception_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr30k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr30k_Xception_fastText",
    "images_processor": "Xception",
    "text_processor": "fastText"
}



config_mixed_coco14_flickr30k_inception_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr30k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr30k_inception_glove",
    "images_processor": "inception",
    "text_processor": "glove"
}
config_mixed_coco14_flickr30k_inception_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr30k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr30k_inception_fastText",
    "images_processor": "inception",
    "text_processor": "fastText"
}




config_mixed_coco14_flickr30k_resnet50_glove = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr30k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr30k_resnet50_glove",
    "images_processor": "resnet50",
    "text_processor": "glove"
}
config_mixed_coco14_flickr30k_resnet50_fastText = {
    "train": {"dataset_name": "coco14", "subset_name": "train"},
    "test": {"dataset_name": "flickr30k", "subset_name": "test"},
    "encode_images": False,
    "save_ix_to_word": False,
    "train_model": True,
    "save_model": True,
    "ixtoword_path": "ixtoword.pkl",
    "wordtoix_path": "wordtoix.pkl",
    "pickles_dir": "/Pickle",
    "encoded_images_test_path": "/encoded_test_images.pkl",
    "encoded_images_train_path": "/encoded_train_images.pkl",
    "model_save_dir": "/model_weights",
    "model_save_path": "/model_Base_3_Batch_Komninos.h5",
    "data_name": "mixed_coco14_flickr30k_resnet50_fastText",
    "images_processor": "resnet50",
    "text_processor": "fastText"
}
