#### Config
data_path = "/home2/data/"
config_flickr8k = {"images_path": data_path + "images/flickr8k/Images/",
           "train_images_path": data_path + "images/flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt", # file conatains the names of images to be used in train data
           "test_images_path": data_path + "images/flickr8k/Flickr8k_text/Flickr_8k.testImages.txt",
           "train_path": data_path + "images/flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt", 
           "token_path": data_path + "images/flickr8k/Flickr8k_text/Flickr_8k.token.txt",
          "word_embedings_path": data_path + "images/glove/glove.6B.200d.txt",
          "embedings_dim": 199,
           "train_model": False, # True if you intend to train the model,
           "save_model": True,
           "encode_images": False,
           "save_ix_to_word": False,
           "ixtoword_path": "./flickr8k/Pickle/ixtoword.pkl",
           "wordtoix_path": "./flickr8k/Pickle/wordtoix.pkl",
           "pretrained_model_path": "",
           "encoded_images_test": "./flickr8k/Pickle/encoded_test_images.pkl",
           "encoded_images_train": "./flickr8k/Pickle/encoded_train_images.pkl",
           "preprocess_descriptions": False,
           "preprocessed_descriptions_save_path": "./flickr8k/descriptions.txt",
           "pretrained_model_path": "",
           "report_path": "",
           "use_lemma": False,
           "spacy_lemma_model": "pl_spacy_model",
           "lstm_model_save_dir": "./flickr8k/model_weights/",
           "lstm_model_save_path": "./flickr8k/model_weights/model_Base_3_Batch_Komninos.h5",
           "results_directory":"results",
           "coco-caption_path":"./coco-caption",
           "data_name":"flickr8k",
           }
config_flickr30k = {"images_folder": data_path + "images/flickr30k/Images",
                 "images_names_path": data_path + "images/flickr30k/karpathy/f30ktalk.json",
                 "token_path": data_path + "images/flickr30k/karpathy/dataset_flickr30k.json",
          "word_embedings_path": data_path + "images/glove/glove.6B.200d.txt",
#           "word_embedings_path": "/home2/data/images/glove/komninos_english_embeddings",
          "embedings_dim": 199,
           #"embedings_dim": 200, # Polish - 100
           "train_model": False, # True if you intend to train the model,
           "save_model": True,
           "encode_images": False,
           "save_ix_to_word": False,
           "ixtoword_path": "./flickr30k/Pickle/ixtoword.pkl",
           "wordtoix_path": "./flickr30k/Pickle/wordtoix.pkl",
           "pretrained_model_path": "",
           "encoded_images_test": "./flickr30k/Pickle/encoded_test_images.pkl",
           "encoded_images_train": "./flickr30k/Pickle/encoded_train_images.pkl",
           "preprocess_descriptions": False,
           "preprocessed_descriptions_save_path": "./flickr30k/descriptions.txt",
           "use_lemma": False,
           "spacy_lemma_model": "pl_spacy_model",
           "report_path": "",
           "lstm_model_save_dir": "./flickr30k/model_weights/",
           "lstm_model_save_path": "./flickr30k/model_weights/model_Base_3_Batch_Komninos.h5",
           "results_directory":"results",
           "coco-caption_path":"./coco-caption",
           "data_name": "flickr30k" 
           }


config_coco14 = {"images_folder": data_path + "images/coco2014",
                 "images_names_path": data_path + "images/coco2014/karpathy/cocotalk.json",
                 "token_path": data_path + "images/coco2014/karpathy/dataset_coco.json",
          "word_embedings_path": data_path + "images/glove/glove.6B.200d.txt",
#           "word_embedings_path": "/home2/data/images/glove/komninos_english_embeddings",
          "embedings_dim": 199,
           #"embedings_dim": 200, # Polish - 100
           "train_model": False, # True if you intend to train the model,
           "save_model": False,
           "encode_images": False,
           "save_ix_to_word": False,
           "ixtoword_path": "./coco14/Pickle/ixtoword.pkl",
           "wordtoix_path": "./coco14/Pickle/wordtoix.pkl",
           "pretrained_model_path": "",
           "encoded_images_test": "./coco14/Pickle/encoded_test_images.pkl",
           "encoded_images_train": "./coco14/Pickle/encoded_train_images.pkl",
           "preprocess_descriptions": False,
           "preprocessed_descriptions_save_path": "./coco14/descriptions.txt",
           "use_lemma": False,
           "spacy_lemma_model": "pl_spacy_model",
           "report_path": "",
           "lstm_model_save_dir": "./coco14/model_weights/",
           "lstm_model_save_path": "./coco14/model_weights/model_Base_3_Batch_Komninos.h5",
           "coco-caption_path":"./coco-caption",
           "results_directory":"results",
           "data_name": "coco14" 
           }
config_coco17 = {"images_folder": data_path + "images/coco2014",
                 "images_names_path": data_path + "images/coco2017/annotations/cocotalk.json",
                 "token_path": data_path + "images/coco2017/annotations/dataset_coco.json",
          "word_embedings_path": data_path + "images/glove/glove.6B.200d.txt",
          "embedings_dim": 199,
           "train_model": True, # True if you intend to train the model,
           "save_model": True,
           "encode_images": False,
           "save_ix_to_word": False,
           "ixtoword_path": "./coco17/Pickle/ixtoword.pkl",
           "wordtoix_path": "./coco17/Pickle/wordtoix.pkl",
           "pretrained_model_path": "",
           "encoded_images_test": "./coco17/Pickle/encoded_test_images.pkl",
           "encoded_images_train": "./coco17/Pickle/encoded_train_images.pkl",
           "preprocess_descriptions": False,
           "preprocessed_descriptions_save_path": "./coco17/descriptions.txt",
           "use_lemma": False,
           "spacy_lemma_model": "pl_spacy_model",
           "report_path": "",
           "lstm_model_save_dir": "./coco17/model_weights/",
           "lstm_model_save_path": "./coco17/model_weights/model_Base_3_Batch_Komninos.h5",
           "coco-caption_path":"./coco-caption",
           "results_directory":"results",
           "data_name": "coco17" 
           }
config_aide = {"images_path": data_path + "images/flickr8k/Images/",
           "train_images_path": data_path + "images/aide/aide_text/aide.trainImages.txt", # file conatains the names of images to be used in train data
           "test_images_path": data_path + "images/aide/aide_text/aide.testImages.txt",
           "train_path": data_path + "images/aide/aide_text/aide.trainImages.txt", 
           "token_path": data_path + "images/aide/aide_text/aide.token.txt",
          "word_embedings_path": data_path + "images/glovePL/glove_100_3_polish.txt",
           "embedings_dim": 99, # Polish - 100
           "train_model": True, # True if you intend to train the model,
           "save_model": True,
           "encode_images": False,
           "save_ix_to_word": True,
           "ixtoword_path": "./aide/Pickle/ixtoword.pkl",
           "wordtoix_path": "./aide/Pickle/wordtoix.pkl",
           "pretrained_model_path": "",
           "encoded_images_test": "./aide/Pickle/encoded_test_images.pkl",
           "encoded_images_train": "./aide/Pickle/encoded_train_images.pkl",
           "preprocess_descriptions": True,
           "preprocessed_descriptions_save_path": "./aide/descriptions.txt",
           "pretrained_model_path": "",
           "report_path": "",
           "use_lemma": False,
           "spacy_lemma_model": "pl_spacy_model",
           "lstm_model_save_dir": "./aide/model_weights/",
           "lstm_model_save_path": "./aide/model_weights/model_Base_3_Batch_Komninos.h5",
           "coco-caption_path":"./coco-caption",
           "results_directory":"results",
           "data_name":"aide",
           }
config_flickr8k_polish = {"images_path": data_path + "images/flickr8k/Images/",
           "train_images_path": data_path + "images/flickr8k_polish/Flickr8k_polish_text/Flickr_8k_polish.trainImages.txt", # file conatains the names of images to be used in train data
           "test_images_path": data_path + "images/flickr8k_polish/Flickr8k_polish_text/Flickr_8k_polish.testImages.txt",
           "train_path": data_path + "images/flickr8k_polish/Flickr8k_polish_text/Flickr_8k_polish.trainImages.txt", 
           "token_path": data_path + "images/flickr8k_polish/Flickr8k_polish_text/Flickr_8k_polish.token.txt",
          "word_embedings_path": data_path + "images/glovePL/glove_100_3_polish.txt",
           "embedings_dim": 99, # Polish - 100
           "train_model": True, # True if you intend to train the model,
           "save_model": True,
           "encode_images": True,
           "save_ix_to_word": True,
           "ixtoword_path": "./flickr8k_polish/Pickle/ixtoword.pkl",
           "wordtoix_path": "./flickr8k_polish/Pickle/wordtoix.pkl",
           "pretrained_model_path": "",
           "encoded_images_test": "./flickr8k_polish/Pickle/encoded_test_images.pkl",
           "encoded_images_train": "./flickr8k_polish/Pickle/encoded_train_images.pkl",
           "preprocess_descriptions": True,
           "preprocessed_descriptions_save_path": "./flickr8k_polish/descriptions.txt",
           "pretrained_model_path": "",
           "report_path": "",
           "use_lemma": False,
           "spacy_lemma_model": "pl_spacy_model",
           "lstm_model_save_dir": "./flickr8k_polish/model_weights/",
           "lstm_model_save_path": "./flickr8k_polish/model_weights/model_Base_3_Batch_Komninos.h5",
           "coco-caption_path":"./coco-caption",
           "results_directory":"results",
           "data_name":"flickr8k_polish",
           }
config_flickr30k_polish = {"images_path": data_path + "images/flickr30k/Images/",
           "train_images_path": data_path + "images/flickr30k_polish/Flickr30k_polish_text/Flickr_30k_polish.trainImages.txt", # file conatains the names of images to be used in train data
           "test_images_path": data_path + "images/flickr30k_polish/Flickr30k_polish_text/Flickr_30k_polish.testImages.txt",
           "train_path": data_path + "images/flickr30k_polish/Flickr30k_polish_text/Flickr_30k_polish.trainImages.txt", 
           "token_path": data_path + "images/flickr30k_polish/Flickr30k_polish_text/Flickr_30k_polish.token.txt",
          "word_embedings_path": data_path + "images/glovePL/glove_100_3_polish.txt",
           "embedings_dim": 99, # Polish - 100
           "train_model": False, # True if you intend to train the model,
           "save_model": False,
           "encode_images": False,
           "save_ix_to_word": False,
           "ixtoword_path": "./flickr30k_polish/Pickle/ixtoword.pkl",
           "wordtoix_path": "./flickr30k_polish/Pickle/wordtoix.pkl",
           "pretrained_model_path": "",
           "encoded_images_test": "./flickr30k_polish/Pickle/encoded_test_images.pkl",
           "encoded_images_train": "./flickr30k_polish/Pickle/encoded_train_images.pkl",
           "preprocess_descriptions": False,
           "preprocessed_descriptions_save_path": "./flickr30k_polish/descriptions.txt",
           "pretrained_model_path": "",
           "report_path": "",
           "use_lemma": False,
           "spacy_lemma_model": "pl_spacy_model",
           "lstm_model_save_dir": "./flickr30k_polish/model_weights/",
           "lstm_model_save_path": "./flickr30k_polish/model_weights/model_Base_3_Batch_Komninos.h5",
           "coco-caption_path":"./coco-caption",
           "results_directory":"results",
           "data_name":"flickr30k_polish",
           }
config_mixed_flickr8k_30k = {"images_path": data_path + "images/flickr8k/Images/",
           "train_images_path": data_path + "images/flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt", # file conatains the names of images to be used in train data
           "test_images_path": data_path + "images/flickr8k/Flickr8k_text/Flickr_8k.testImages.txt",
           "train_path": data_path + "images/flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt", 
           "token_path": data_path + "images/flickr8k/Flickr8k_text/Flickr_8k.token.txt",
          "word_embedings_path": data_path + "images/glove/glove.6B.200d.txt",
          "embedings_dim": 199,
           "train_model": False, # True if you intend to train the model,
           "save_model": True,
           "encode_images": False,
           "save_ix_to_word": False,
           "ixtoword_path": "./flickr8k/Pickle/ixtoword.pkl",
           "wordtoix_path": "./flickr8k/Pickle/wordtoix.pkl",
           "pretrained_model_path": "",
           "encoded_images_test": "./flickr8k/Pickle/encoded_test_images.pkl",
           "encoded_images_train": "./flickr8k/Pickle/encoded_train_images.pkl",
           "preprocess_descriptions": False,
           "preprocessed_descriptions_save_path": "./flickr8k/descriptions.txt",
           "pretrained_model_path": "",
           "report_path": "",
           "use_lemma": False,
           "spacy_lemma_model": "pl_spacy_model",
           "lstm_model_save_dir": "./flickr8k/model_weights/",
           "lstm_model_save_path": "./flickr8k/model_weights/model_Base_3_Batch_Komninos.h5",
           "results_directory":"results",
           "coco-caption_path":"./coco-caption",
           "data_name":"flickr8k",
           }