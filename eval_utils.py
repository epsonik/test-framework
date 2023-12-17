from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import sys
import time
from time import time
from numpy import argmax, argsort
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from config import general
import csv


def calculate_results(expected, results, config, model_name):
    """
    Method to evaluate image captioning model by calculating scores:
     "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"m "METEOR", "ROUGE_L", "CIDEr", "WMD", "SPICE"

    Parameters
    ----------
    expected: dict
        Dictionary with ground truth captions, identidied by image_id
    results: dict
        Dictionary with predicted captions, identified by image_id
    config
        Configuration of training and testing
    Returns
    -------
    calculated_metrics
        Result of the metrics: "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"m "METEOR", "ROUGE_L", "CIDEr", "WMD", "SPICE"

    """
    sys.path.append(general["coco-caption_path"])
    from pycocoevalcap.eval_any import COCOEvalCap
    # Load expected captions(ground truth from dataset) and results(predicted captions for specific image)
    # to the evaluation framework
    cocoEvalObj = COCOEvalCap(expected, results)
    # Evaluate
    cocoEvalObj.evaluate()
    calculated_metrics = {}
    # Store metrics  values in dictionary by metrics names
    for metric, score in cocoEvalObj.eval.items():
        calculated_metrics[metric] = score
    print(calculated_metrics)
    print("Calculating final results")
    imgToEval = cocoEvalObj.imgToEval
    for p in results:
        print(imgToEval)
        image_id, caption = p, results[p][0]['caption']
        imgToEval[image_id]['caption'] = caption
        imgToEval[image_id]['ground_truth_captions'] = [x['caption'] for x in expected[p]]

    model_result_dir = general["results_directory"] + "/" + config["data_name"]
    if not os.path.isdir(model_result_dir):
        os.makedirs(model_result_dir)

    evaluation_results_save_path = os.path.join(model_result_dir, model_name.replace(".h5", '') + '.json')
    print("Results saved to ")
    print(evaluation_results_save_path)
    # Path to save evaluation results
    with open(evaluation_results_save_path, 'w') as outfile:
        json.dump(
            {'overall': calculated_metrics, 'dataset_name': model_name.split('.')[0], 'imgToEval': imgToEval},
            outfile)
    return calculated_metrics


def beam_search_pred(photo, model, wordtoix, ixtoword, max_length, k_beams=5, log=True):
    start = [wordtoix[general["START"]]]

    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            sequence = pad_sequences([s[0]], maxlen=max_length).reshape(
                (1, max_length))  # sequence of most probable words
            # based on the previous steps
            preds = model([photo, sequence])
            word_preds = argsort(preds[0])[-k_beams:]  # sort predictions based on the probability, then take the last
            # K_beams items. words with the most probs
            # Getting the top <K_beams>(n) predictions and creating a
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                if log:
                    prob += np.log(preds[0][w])  # assign a probability to each K words4
                else:
                    prob += preds[0][w]
                temp.append([next_cap, prob])
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])

        # Getting the top words
        start_word = start_word[-k_beams:]

    start_word = start_word[-1][0]
    captions_ = [ixtoword[i] for i in start_word]

    final_caption = []

    for i in captions_:
        if i != general["STOP"]:
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption


def greedySearch(photo, model, wordtoix, ixtoword, max_length):
    """
    Method to put ground truth captions and results to the structure accepted by evaluation framework

    Parameters
    ----------
    encoding_test: str
        Path to the results directory
    test_captions_mapping
        Dictionary with keys-image_id's , values -list of ground truth captions for specific image.
    wordtoix
        Dictionary with keys-words , values -id of word
    ixtoword
        Dictionary with keys-id of words , values -words
    max_length
        Max number of words in caption on dataset
    model
        Image captioning model to predict captions
    Returns
    -------
    expected: dict
        Dictionary with ground truth captions, identidied by image_id
    results: dict
        Dictionary with predicted captions, identified by image_id

    """
    in_text = general["START"]
    for i in range(max_length):
        # Get previously generated sequence
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        print("photo")
        print(type(photo))
        # Pad sewuences to the maximum length
        sequence = pad_sequences([sequence], maxlen=max_length)
        print("sequence")
        print(type(sequence))
        # Predict sequence with the learned model
        yhat = model([photo, sequence])
        # Get word with the highest propability
        yhat = argmax(yhat)
        # Transform index of word to the word by previously created dictionary
        word = ixtoword[yhat]
        in_text += ' ' + word
        # When we achieve STOP word, sequence is generated
        if word == general["STOP"]:
            break
    final = in_text.split()
    # remove start and stop words
    final = final[1:-1]
    # Create sencece by joining tokens
    final = ' '.join(final)
    return final


def prepare_for_evaluation(encoding_test, test_captions_mapping, wordtoix, ixtoword, max_length, model,
                           images_processor):
    """
    Method to put ground truth captions and results to the structure accepted by evaluation framework

    Parameters
    ----------
    encoding_test: str
        Path to the results directory
    test_captions_mapping
        Dictionary with keys-image_id's , values -list of ground truth captions for specific image.
    wordtoix
        Dictionary with keys-words , values -id of word
    ixtoword
        Dictionary with keys-id of words , values -words
    max_length
        Max number of words in caption on dataset
    model
        Image captioning model to predict captions
    images_processor

    Returns
    -------
    expected: dict
        Dictionary with ground truth captions, identidied by image_id
    results: dict
        Dictionary with predicted captions, identified by image_id

    """
    # Get all image-ids from test dataset
    test_pics = list(encoding_test.keys())
    expected = dict()
    results = dict()
    print("Preparing for evaluation")
    # calculation of metrics for test images dataset
    index = 0
    for j in range(0, len(test_pics)):
        image_id = test_pics[j]
        expected[image_id] = []
        if images_processor == 'vgg16' or images_processor == 'vgg19':
            image = encoding_test[image_id].reshape((1, 4096))
        elif images_processor == 'resnet152V2':
            image = encoding_test[image_id].reshape((1, 2048))
        elif images_processor == 'denseNet121':
            image = encoding_test[image_id].reshape((1, 1024))
        elif images_processor == 'mobileNet':
            image = encoding_test[image_id].reshape((1, 1000))
        elif images_processor == 'mobileNetV2':
            image = encoding_test[image_id].reshape((1, 1280))
        elif images_processor == 'denseNet201':
            image = encoding_test[image_id].reshape((1, 1920))
        else:
            image = encoding_test[image_id].reshape((1, 2048))

        # Put ground truth captions to the structure accepted by evaluation framework.
        for desc in test_captions_mapping[image_id]:
            expected[image_id].append({"image_id": image_id, "caption": desc})
        # Predict captions

        st = time()
        generated = greedySearch(image, model, wordtoix, ixtoword, max_length)
        et = time()
        # get the execution time
        elapsed_time = et - st
        # print('Execution time:', elapsed_time*1000, 'miliseconds')

        # get the execution time
        # Put predicted captions to the structure accepted by evaluation framework.
        results[image_id] = [{"image_id": image_id, "caption": generated, "time": elapsed_time}]
        if index % 100 == 0:
            print("Processed:")
            print(index)
        index += 1
    return expected, results


def generate_report(results_path):
    """
    Method to generate summary of the test results. Made from files in the results directory.

    Parameters
    ----------
    results_path: str
        Path to the results directory
    Returns
    -------
        CSV file with summary of the results.

    """
    # Names of the evaluation metrics
    header = ["config_name", "loss", "epoch", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr",
              "SPICE", "WMD"]
    print(f'\n Final results saved to final_results.csv')
    all_results = []
    # iterate over all files in results directory
    for x in os.listdir(results_path):
        # use just .json files
        if x.endswith(".json"):
            # Load data from file with results particular for configuaration
            results_for_report = json.load(open(os.path.join(results_path, x), 'r'))
            # Add column with the configuration name to name the specific results.
            config_name = x.replace(".json", '')
            b = config_name.split("-")
            results_for_report["overall"]["config_name"] = config_name
            results_for_report["overall"]["loss"] = b[2]
            results_for_report["overall"]["epoch"] = b[1]
            # Save the results to the table to save it in the next step
            all_results.append(results_for_report["overall"])
    # Save final csv file

    with open(results_path + "/final_results.csv", 'w') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_results)
