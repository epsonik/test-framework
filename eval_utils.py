from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import sys

import numpy as np
from keras.preprocessing.sequence import pad_sequences

from config import general


def calculate_results(expected, results, config):
    sys.path.append(general["coco-caption_path"])
    from pycocoevalcap.eval_any import COCOEvalCap
    cocoEval = COCOEvalCap(expected, results)
    cocoEval.evaluate()
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score
    print(out)
    print("Calculating final results")
    imgToEval = cocoEval.imgToEval
    for p in results:
        print(imgToEval)
        image_id, caption = p, results[p][0]['caption']
        imgToEval[image_id]['caption'] = caption
        imgToEval[image_id]['ground_truth_captions'] = [x['caption'] for x in expected[p]]

    cache_path = os.path.join(general["results_directory"], config["data_name"] + '.json')
    print("Results saved to ")
    print(cache_path)
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'dataset_name': config["test"]["dataset_name"], 'imgToEval': imgToEval}, outfile)
    return out


def greedySearch(photo, model, wordtoix, ixtoword, max_length):
    in_text = general["START"]
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == general["STOP"]:
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


def prepare_for_evaluation(encoding_test, data, model):
    from tqdm.notebook import tqdm
    test_pics = list(encoding_test.keys())
    expected = {}
    results = {}
    print("Preparing for evaluation")
    # calculation of metrics for test images dataset
    index = 0
    for j in tqdm(range(0, len(test_pics))):
        image_id = test_pics[j]
        expected[image_id] = []
        image = encoding_test[image_id].reshape((1, 2048))
        for desc in data.test_captions_mapping[image_id]:
            expected[image_id].append({"image_id": image_id, "caption": desc})
        generated = greedySearch(image, model, data.wordtoix, data.ixtoword, data.max_length)
        results[image_id] = [{"image_id": image_id, "caption": generated}]
        if index % 100 == 0:
            print("Processed:")
            print(index)
        index += 1
    return expected, results


def generate_report(results_path):

    import csv
    header = ["config_name", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE", "WMD"]
    print(f'\n Final results saved to final_results.csv')
    overall = []
    for x in os.listdir(results_path):
        if x.endswith(".json"):
            results_for_report = json.load(open("./" + results_path + "/" + x, 'r'))
            results_for_report["overall"]["config_name"] = x.split(".")[0]
            overall.append(results_for_report["overall"])
    with open("./" + results_path + "/final_results.csv", 'w') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(overall)
