import os
import json
from pycocoevalcap.eval import COCOEvalCap, Bleu, Meteor, Rouge, Cider, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO
import unicodedata

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

import logging

eval_logger = logging.getLogger("lmms-eval")

dir_name = os.path.dirname(os.path.abspath(__file__))

xm100_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]  # , "SPICE"]


def xm100_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def xm100_doc_to_text(doc):
    return f"Provide a one-sentence caption for the provided image."


def xm100_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0] if len(result) > 0 else ""
    # question_id = doc["question_id"]
    # The question id in our dataset is the image file itself
    # image_id = int(question_id.split("_")[-1].split(".")[0])
    # id = doc["id"]
    image_id = doc["image_id"]
    # convert str to int, replace a-z with 1-26
    int_id = ""
    for c in image_id:
        if c.isalpha():
            int_id += str(ord(c) - 96)
        else:
            int_id += c
    
    id = int(int_id)

    

    data_dict = {"answer": [doc["caption"]], "pred": pred, "image_id": id}

    return {f"xm100_{metric}": data_dict for metric in xm100_METRICS}


def normalize(result, args):
    caption = result["pred"]
    if 'xm100_zh' in args.tasks:
        from spacy.lang.zh import Chinese
        chinese = Chinese() #.from_config({"nlp": {"tokenizer": {"segmenter": "jieba"}}})
        caption = "".join([word.text for word in chinese(caption)])
    if 'xm100_jp' in args.tasks:
        from spacy.lang.ja import Japanese
        japanese = Japanese()
        caption = "".join([word.text for word in japanese(caption)])
    if 'xm100_th' in args.tasks:
        from spacy.lang.th import Thai
        thai = Thai()
        caption = "".join([word.text for word in thai(caption)])
    caption = unicodedata.normalize("NFC", caption)
    result["pred"] = caption
    return result

def xm100_aggregation_result(results, metric, args):
    scorers = [(Bleu(4), "Bleu_1"), (Bleu(4), "Bleu_2"), (Bleu(4), "Bleu_3"), (Bleu(4), "Bleu_4"), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr"), (Spice(), "SPICE")]
    scorers_dict = {s[1]: s for s in scorers}

    stored_results = []
    # In order to make the coco eval tools to successfully create index
    # We need at least two dict in the dataset
    # 'annotation' and 'images'
    # 'annotation' exactly reproduce the original annotation
    # 'images' however only need the image id which is contained in the file name
    dataset = {"annotations": [], "images": []}
    idx = 0
    for result in results:
        result = normalize(result, args)
        stored_results.append({"image_id": int(result["image_id"]), "caption": result["pred"]})
        for a in result["answer"]:
            dataset["annotations"].append({"image_id": int(result["image_id"]), "caption": a, "id": idx})
            idx += 1
        dataset["images"].append({"id": result["image_id"]})

    coco = COCO()
    # Manually create index here
    coco.dataset = dataset
    coco.createIndex()

    coco_result = coco.loadRes(stored_results)
    coco_eval = COCOEvalCap(coco, coco_result)

    imgIds = coco_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = coco_eval.coco.imgToAnns[imgId]
        res[imgId] = coco_eval.cocoRes.imgToAnns[imgId]

    eval_logger.info("tokenization...")

    # if 'xm100_zh' not in args.tasks and 'xm100_jp' not in args.tasks and 'xm100_th' not in args.tasks:
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    eval_logger.info(f"Computing {metric} scores...")

    score, scores = scorers_dict[metric][0].compute_score(gts, res)
    # When metric is one of the Bleu, score will be a list
    if type(score) == list:
        n = int(metric.split("_")[-1])
        score = score[n - 1]

    path = generate_submission_file("xm100_captions_val_alg_results.json", args)
    if not os.path.exists(path):
        eval_logger.info("Storing prediction that can be submitted to the server ...")
        with open(path, "w") as f:
            json.dump(stored_results, f, indent=4)

    return score


def xm100_bleu4(results, args):
    return xm100_aggregation_result(results, "Bleu_4", args)


def xm100_bleu3(results, args):
    return xm100_aggregation_result(results, "Bleu_3", args)


def xm100_bleu2(results, args):
    return xm100_aggregation_result(results, "Bleu_2", args)


def xm100_bleu1(results, args):
    return xm100_aggregation_result(results, "Bleu_1", args)


def xm100_meteor(results, args):
    return xm100_aggregation_result(results, "METEOR", args)


def xm100_rougel(results, args):
    return xm100_aggregation_result(results, "ROUGE_L", args)


def xm100_cider(results, args):
    return xm100_aggregation_result(results, "CIDEr", args)


def xm100_spice(results, args):
    return xm100_aggregation_result(results, "SPICE", args)


def xm100_test_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case coco_passthrough), value: metric value
    """
    # question_id = doc["question_id"]
    # # The question id in our dataset is the image file itself
    # image_id = int(question_id.split("_")[-1].split(".")[0])
    image_id = doc["image_id"]
    # convert str to int, replace a-z with 1-26
    int_id = ""
    for c in image_id:
        if c.isalpha():
            int_id += str(ord(c) - 96)
        else:
            int_id += c
    
    id = int(int_id)
    return {"xm100_passthrough": {"pred": result, "image_id": id}}


def xm100_test_aggregation_result(results, args):
    stored_results = []
    for result in results:
        result = normalize(result, args)
        stored_results.append({"image_id": int(result["image_id"]), "caption": result["pred"]})

    path = generate_submission_file("xm100_captions_test_alg_results.json", args)
    eval_logger.info("Storing prediction that can be submitted to the server ...")
    with open(path, "w") as f:
        json.dump(stored_results, f, indent=4)

    eval_logger.info(f"Your test result has been stored in to {path}. Make sure you also have the val result stored to submit to the server on https://codalab.lisn.upsaclay.fr/competitions/7404#participate.")
