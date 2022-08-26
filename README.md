## Environment setup

Copy folder`coco-caption` from path `/home2/data` to project root directory
create virtual environment from file
```
conda create --name framework --file requirements.txt
```
Add virtual environment `framework` to jupyter
```
ipython kernel install --user --name=framework
```
### How to use jupyter Notebook via ssh:
How to connect local graphic jupyter notebook to remote jupyter server with virtual conda environment via ssh.

WARNING:
If remote server is private remember to launch VPN.

Launch terminal and run command:
```
ssh -N -L localhost:54321:localhost:54321 wisla@10.44.27.164
(password)
```
Launch second terminal and run command:

```
ssh wisla@10.44.27.164
(password)
```
```

```
## Configuration file
File In `config.py`  is a configuration file that allows to automatically load data for testing framework.


### train_images_path

Path to the file with ids of images from training Dataset

### test_images_path

Path to the file with ids of images from test Dataset

### token_path

Path to the file with ids of images and captions

### word_embedings_path"

Path to the file with GLOVE word embeddings

### embedings_dim

Size of the GLOVE embeddings vector: Polish 199, English 299

### train_model

if __True__ model from the current training will be saved in path __"lstm_model_save_path"__ 

if __False__ model will be loaded from __"lstm_model_save_path"__ 

### save_model

if __True__ model from the current training will be saved in path __"lstm_model_save_path"__ 

if __False__ model will be loaded from __"lstm_model_save_path"__ 

### encode_images

if __True__ images will be encoded and saved under path __"encoded_images_test"__ from config file

if __False__ encoded images will be loaded from path __"lstm_model_save_path"__ from config file

### save_ix_to_word

if __True__ images will be encoded and saved under path __"ixtoword_path"__
if __False__ model will be loaded from __"lstm_model_save_path"__ 

### ixtoword_path

Path of the dictionary that maps indexes of words to words

### wordtoix_path

Path of the dictionary that maps words to indexes 

### encoded_images_test

Path to file with encoded images from test set

### encoded_images_train

Path to file with encoded images from train set

### preprocess_descriptions

if __True__ descriptions will be encoded and saved under path __"preprocessed_descriptions_save_path"__

if __False__ descriptions will be loaded from __"preprocessed_descriptions_save_path"__ 

### preprocessed_descriptions_save_path

Path to the directory, where descriptions translated to the dictionary form will be stored

### lstm_model_save_dir

Name of the directory to store model weights

### lstm_model_save_path

Name of the file to store model

### results_directory

Name of the directory, where evaluation file iss stored.

### coco-caption_path

Path to the directory with coco-caption library

### data_name

Name of the data that is used fe. Flickr8k, flickr30k This name is used to store all Pickle files and preprocessed file.

## How to use
To run specific Dataset we need to pass name of configuration from `config.py` to function:
```
data = DataLoader(name_of_configuration)
example
data = DataLoader(config_flickr30k_polish)
```
Supported configurations(Datasets) :
| Name of configuration| Dataset |
| --- | ----------- |
| config_flickr8k | Flickr8k |
| config_flickr30k | Flickr30k |
|config_coco14| COCO Dataset 2014|
|config_coco17| COCO Dataset 2017|
|config_aide|AiDE Dataset Polish|
| config_flickr8k_polish | Flickr8k in Polish|
| config_flickr30k_polish | Flickr30k in Polish|


