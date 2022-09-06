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
    eval_res = cocoEval.evaluate()
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in results:
        image_id, caption = p, results[p][0]['caption']
        imgToEval[image_id]['caption'] = caption
        imgToEval[image_id]['ground_truth_captions'] = [x['caption'] for x in expected[p]]

    if not os.path.isdir("./" + general["results_directory"]):
        os.makedirs("./" + general["results_directory"])
    cache_path = os.path.join(general["results_directory"], config["data_name"] + '.json')
    print(cache_path)
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)
    return out


def greedySearch(photo, model, wordtoix, ixtoword, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
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
    # calculation of metrics for test images dataset
    for j in tqdm(range(0, len(test_pics))):
        image_id = test_pics[j]
        expected[image_id] = []
        image = encoding_test[image_id].reshape((1, 2048))
        for desc in data.test_captions_mapping[image_id]:
            expected[image_id].append({"image_id": image_id, "caption": desc})
        generated = greedySearch(image, model, data.wordtoix, data.ixtoword, data.max_length)
        results[image_id] = [{"image_id": image_id, "caption": generated}]
    return expected, results
