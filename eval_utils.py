from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys

def calculate_results(expected, results, config):
    sys.path.append(config["coco-caption_path"])
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

    if not os.path.isdir("./"+ config["results_directory"]):
        os.makedirs("./"+config["results_directory"])
    cache_path = os.path.join(config["results_directory"], config["data_name"] + '.json')
    print(cache_path)
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)
    return out