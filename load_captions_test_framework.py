import json
import os
import pandas as pd

# Opening JSON file and loading the data
# into the variable data
base_path = os.path.abspath(os.path.dirname(__file__))  # path to this project

k_1 = os.path.join(base_path, "mixed_coco14_coco14_Xception_glove_concatenate_dense512", "k-1")
k_2 = os.path.join(base_path, "mixed_coco14_coco14_Xception_glove_concatenate_dense512", "k-2")
k_5 = os.path.join(base_path, "mixed_coco14_coco14_Xception_glove_concatenate_dense512", "k-5")

result_files_for_beams = {"k-1": k_1, "k-2": k_2, "k-5": k_5}
epoch_list = list()
beam_list = list()
img_id_list = list()
bleu_1_list = list()
bleu_2_list = list()
bleu_3_list = list()
bleu_4_list = list()
rouge_list = list()
cider_list = list()
ground_truth_captions_list = list()
caption_list = list()
loss_list = list()

overall_epoch_list = list()
overall_beam_list = list()
overall_bleu_1_list = list()
overall_bleu_2_list = list()
overall_bleu_3_list = list()
overall_bleu_4_list = list()
overall_rouge_list = list()
overall_cider_list = list()
overall_loss_list = list()

for beam, result_path_for_beam in result_files_for_beams.items():
    result_files_for_beam = os.listdir(result_path_for_beam)
    for model_name in result_files_for_beam:
        model_path = os.path.join(result_path_for_beam, model_name)
        if model_name.endswith(".json") and (not model_name.startswith('.')):
            with open(model_path) as json_file:
                data = json.load(json_file)
            img_to_eval = data['imgToEval']
            for img_id in img_to_eval.keys():
                image_data = img_to_eval[img_id]
                epoch = model_name.split("-")[-2]
                epoch_list.append(epoch)
                beam_list.append(beam)
                loss = model_name.split("-")[-1].replace(".json", "")
                loss_list.append(loss)
                img_id_list.append(img_id)
                bleu_1 = image_data["Bleu_1"]
                bleu_1_list.append(bleu_1)
                bleu_2 = image_data["Bleu_2"]
                bleu_2_list.append(bleu_2)
                bleu_3 = image_data["Bleu_3"]
                bleu_3_list.append(bleu_3)
                bleu_4 = image_data["Bleu_4"]
                bleu_4_list.append(bleu_4)
                rouge = image_data["ROUGE_L"]
                rouge_list.append(rouge)
                cider = image_data["CIDEr"]
                cider_list.append(cider)
                ground_truth_captions = image_data["ground_truth_captions"]
                ground_truth_captions_list.append(ground_truth_captions)
                caption = image_data["caption"]
                caption_list.append(caption)

            overall = data['overall']
            epoch = model_name.split("-")[-2]
            overall_epoch_list.append(epoch)
            overall_beam_list.append(beam)

            loss = model_name.split("-")[-1].replace(".json", "")
            overall_loss_list.append(loss)

            bleu_1 = overall["Bleu_1"]
            overall_bleu_1_list.append(bleu_1)
            bleu_2 = overall["Bleu_2"]
            overall_bleu_2_list.append(bleu_2)
            bleu_3 = overall["Bleu_3"]
            overall_bleu_3_list.append(bleu_3)
            bleu_4 = overall["Bleu_4"]
            overall_bleu_4_list.append(bleu_4)
            rouge = overall["ROUGE_L"]
            overall_rouge_list.append(rouge)
            cider = overall["CIDEr"]
            overall_cider_list.append(cider)
cap = pd.DataFrame(
    {"epoch": epoch_list, "beam": beam_list, "loss": loss_list, "image_id": img_id_list, "Bleu_1": bleu_1_list,
     "Bleu_2": bleu_2_list,
     "Bleu_3": bleu_3_list,
     "Bleu_4": bleu_4_list, "ROUGE_L": rouge_list,
     "CIDEr": cider_list,
     "ground_truth_captions": ground_truth_captions_list, "caption": caption_list})

cap.to_csv("test_framework.csv")

overall = pd.DataFrame(
    {"epoch": overall_epoch_list, "beam": overall_beam_list, "loss": overall_loss_list, "Bleu_1": overall_bleu_1_list,
     "Bleu_2": overall_bleu_2_list,
     "Bleu_3": overall_bleu_3_list,
     "Bleu_4": overall_bleu_4_list, "ROUGE_L": overall_rouge_list,
     "CIDEr": overall_cider_list})
overall.to_csv("overall_test_framework.csv")
