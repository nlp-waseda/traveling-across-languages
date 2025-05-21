from collections import defaultdict
import os
import datetime
import json
from typing import Dict, List, Literal
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from datasets import Image as DatasetsImage
from pycocoevalcap.eval import COCOEvalCap, Rouge, Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

import unicodedata
import re

import pandas as pd

import os
from loguru import logger as eval_logger



# Example usage with extended synonyms for English, French, Hindi, Hebrew, Romanian, Thai, and Chinese
synonyms = {
    "true": ["yes", "1", "true", "vrai", "हां", "נכון", "adevărat", "ใช่", "是"],
    "false": ["no", "0", "false", "faux", "नहीं", "לא נכון", "fals", "ไม่ใช่", "否"],
    "yes": ["true", "1", "yes", "oui", "हां", "כן", "da", "ใช่", "是"],
    "no": ["false", "0", "no", "non", "नहीं", "לא", "nu", "ไม่ใช่", "否"],
    # Add more entries as needed for other languages
}



maxm_METRICS = ["rouge_l", "cider", "exact_match", "relaxed_accuracy"]

def maxm_doc_to_visual(doc):
    # This is necessary, for reference check: https://huggingface.co/datasets/floschne/maxm 
    pil_image = DatasetsImage().decode_example(doc["image"])
    return [pil_image.convert("RGB")]

def maxm_doc_to_text(doc, model_specific_prompt_kwargs=None):
    question = doc["question"].strip()
    if model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs.get("pre_prompt", "")
        post_prompt = model_specific_prompt_kwargs.get("post_prompt", "")
        question = f"{pre_prompt}{question}{post_prompt}"
    return question

def maxm_process_results(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    
    pred = result[0] if len(result) > 0 else ""
    image_id = doc["image_id"]
    # convert str to int, replace a-z with 1-26
    int_id = ""
    for c in image_id:
        if c.isalpha():
            int_id += str(ord(c) - 96)
        else:
            int_id += c
    
    id = int(int_id)

    data_dict = {"answer": doc["answers"], "pred": pred, "image_id": id}

    return {f"{metric}": data_dict for metric in maxm_METRICS}

def exact_match_with_multiple_references(predictions, references):
    exact_matches = []
    for pred, ref_list in zip(predictions, references):
        if isinstance(pred, list):
            pred = pred[-1]
        match = pred in [ans for ans in ref_list] #any(pred == ref for ref in ref_list)
        exact_matches.append(match)
    return {"exact_match": 100*sum(exact_matches) / len(exact_matches)}

def preprocess_answer(answer):
    """Preprocess the answer by lowercasing, stripping, and removing punctuation."""
    # Normalize unicode characters
    answer = unicodedata.normalize('NFKD', answer)
    # Lowercase (except for scripts where case doesn't apply, like Chinese)
    answer = answer.lower()
    # Strip whitespace
    answer = answer.strip()
    # Remove punctuation (keep language-specific characters intact)
    answer = re.sub(r'[^\w\s]', '', answer)
    return answer


def is_correct(generated, gold_answers, synonyms=None):
    """Check if the generated answer matches any of the gold answers using relaxed accuracy criteria."""
    generated = preprocess_answer(generated)
    
    for gold in gold_answers:
        gold = preprocess_answer(gold)

        # Check for exact match
        if generated == gold:
            return True
        
        if generated in gold:
            return True

        # # Check if generated answer starts or ends with the gold answer
        if generated.startswith(gold) or generated.endswith(gold):
            return True

        # Check for synonyms (e.g., yes/no, true/false)
        if synonyms and generated in synonyms.get(gold, []):
            return True

    return False


def generated_label_classification_evaluation(
    gold_labels: List[str] | List[List[str]],
    pred_labels: List[str],
    vqa_post_process: bool = True,
    bool_to_yes_no: bool = True,
    entailment_to_yes_no_maybe: bool = False,
    remove_trailing_period: bool = True,
) -> Dict[str, float]:
    single_answers = isinstance(gold_labels[0], str)
    print(f"Evaluation with {single_answers=}")

    df = pd.DataFrame(
        {
            "gold": gold_labels,
            "pred": pred_labels,
        }
    )

    if single_answers:
        df["gold"] = df["gold"].apply(lambda x: str(x).strip())
        if remove_trailing_period:
            df["pred"] = df["pred"].apply(lambda x: str(x).rstrip("."))
    else:
        df["gold"] = df["gold"].apply(lambda x: [str(ans).strip() for ans in x])
        if remove_trailing_period:
            df["pred"] = df["pred"].apply(lambda x: [str(ans).rstrip(".") for ans in x])

    if entailment_to_yes_no_maybe:

        def _entailment_to_yes_no_maybe(x) -> Literal["yes", "no", "maybe"]:
            if x.lower() in ["yes", "no", "maybe"]:
                return x.lower()
            elif x.lower() in ["true", "false"]:
                return "yes" if x.lower() == "true" else "no"
            elif x.lower() in ["1", "0"]:
                return "yes" if x.lower() == "1" else "no"
            elif x.lower() in ["entailment", "contradiction", "neutral"]:
                if x.lower() == "entailment":
                    return "yes"
                elif x.lower() == "contradiction":
                    return "no"
                else:
                    return "maybe"
            else:
                return x

        df["pred"] = df["pred"].apply(_entailment_to_yes_no_maybe)
        if single_answers:
            df["gold"] = df["gold"].apply(_entailment_to_yes_no_maybe)
        else:
            df["gold"] = df["gold"].apply(
                lambda x: [_entailment_to_yes_no_maybe(ans) for ans in x]
            )

    if bool_to_yes_no:

        def map_bool_to_yes_no(x):
            if isinstance(x, bool):
                return "yes" if x else "no"
            elif isinstance(x, str):
                if x.lower() == "true":
                    return "yes"
                elif x.lower() == "false":
                    return "no"
            return x

        df.pred = df.pred.apply(lambda x: str(map_bool_to_yes_no(x)).strip())
        if single_answers:
            df.gold = df.gold.apply(lambda ans: str(map_bool_to_yes_no(ans)).strip())
        else:
            df.gold = df.gold.apply(
                lambda answers: [map_bool_to_yes_no(ans) for ans in answers]
            )

    df["pred_post_processed"] = vqa_clean(df["pred"].tolist())

    # print(df.head())

    scores: Dict[str, float] = {}
    if single_answers:
        scores["acc"] = (df["gold"] == df["pred"]).mean()
        scores["relaxed_acc"] = df.apply(
            lambda x: x["pred"].startswith(x["gold"]) or x["pred"].endswith(x["gold"]),
            axis=1,
        ).mean()

        if vqa_post_process:
            post_proc_acc = (df["gold"] == df["pred_post_processed"]).mean()
            post_proc_relaxed_acc = df.apply(
                lambda x: x["pred_post_processed"].startswith(x["gold"])
                or x["pred_post_processed"].endswith(x["gold"]),
                axis=1,
            ).mean()
            scores["acc_post_processed"] = post_proc_acc
            scores["relaxed_acc_post_processed"] = post_proc_relaxed_acc
    else:
        scores["acc"] = df.apply(lambda x: x["pred"] in x["gold"], axis=1).mean()
        scores["relaxed_acc"] = df.apply(
            lambda x: any(
                x["pred"].startswith(ans) or x["pred"].endswith(ans)
                for ans in x["gold"]
            ),
            axis=1,
        ).mean()

        if vqa_post_process:
            scores["acc_post_processed"] = df.apply(
                lambda x: any(x["pred_post_processed"] == ans for ans in x["gold"]),
                axis=1,
            ).mean()
            scores["relaxed_acc_post_processed"] = df.apply(
                lambda x: any(
                    x["pred_post_processed"].startswith(ans)
                    or x["pred_post_processed"].endswith(ans)
                    # or x["pred_post_processed"] in ans
                    for ans in x["gold"]
                ),
                axis=1,
            ).mean()
            # print(df.head())

    return scores



# adapted from https://github.com/salesforce/LAVIS/blob/main/lavis/common/vqa_tools/vqa_eval.py
def vqa_clean(labels: List[str]):
    manualMap = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    articles = ["a", "an", "the"]
    periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
    commaStrip = re.compile(r"(\d)(,)(\d)")
    punct = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]
    contractions = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(commaStrip, inText) is not None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = manualMap.setdefault(word, word)
            if word not in articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in contractions:
                outText[wordId] = contractions[word]
        outText = " ".join(outText)
        return outText

    cleaned_labels = []
    for label in labels:
        label = label.replace("\n", "").replace("\t", "").strip()
        label = processPunctuation(label)
        label = processDigitArticle(label)
        label = ''.join(label)
        label = label.strip("'")
        cleaned_labels.append(label)
    # print(cleaned_labels)
    return cleaned_labels


def relaxed_accuracy_metric(generated_answers, gold_answers, synonyms=None):
    """Calculate the relaxed accuracy for a list of generated and gold answers."""
    correct = 0
    for gen_ans, gold_ans in zip(generated_answers, gold_answers):
        if is_correct(gen_ans, gold_ans, synonyms=synonyms):
            correct += 1
    return 100*correct / len(generated_answers)


def maxm_aggregate_results_v2(results, metric, args):

    print('Currently in Metric: ', metric)

    if metric == "exact_match":

        preds = []
        for r in results:
            if isinstance(r["pred"], list) and len(r["pred"]) > 1:
                preds.append(r["pred"][-1])
            else:
                preds.append(r["pred"])
            # r["pred"][-1] for r in results if len(r["pred"])>1 else r["pred"]]
        res = [r["answer"] for r in results]

        return exact_match_with_multiple_references(preds, res)['exact_match']
    
    if metric == 'relaxed_accuracy':
        preds = []
        for r in results:
            if isinstance(r["pred"], list) and len(r["pred"]) > 1:
                preds.append([r["pred"][-1]])
            else:
                preds.append([r["pred"]])
            # r["pred"][-1] for r in results if len(r["pred"])>1 else r["pred"]]
        res = [r["answer"] for r in results]
        # print('Preds: ', preds[:5])
        # print('Res: ', res[:5])
        # return relaxed_accuracy_metric(preds, res, synonyms)
        # return relaxed_accuracy_metric(preds, res)
        return 100*generated_label_classification_evaluation(res, preds)['relaxed_acc_post_processed']


    scorers = [(Rouge(), "rouge_l"), (Cider(), "cider")]
    scorers_dict = {s[1]: s for s in scorers}

    # Prepare ground truth and prediction data for COCO format
    ground_truths = []
    predictions = []
    idx = 0
    for item in results:
        image_id = item['image_id']
        for answer in item['answer']:
            ground_truths.append({
                'image_id': image_id,
                'caption': answer,
                'id': idx
            })
            idx += 1
        predictions.append({
            'image_id': image_id,
            'caption': item['pred']
        })

    # Create COCO-like annotations for ground truth
    coco = COCO()
    coco.dataset = {'images': [{'id': item['image_id']} for item in results], 'annotations': ground_truths}
    coco.createIndex()

    # Create a fake results file
    coco_result = coco.loadRes(predictions)

    # Evaluation setup
    coco_eval = COCOEvalCap(coco, coco_result)
    imgIds = coco_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = coco_eval.coco.imgToAnns[imgId]
        res[imgId] = coco_eval.cocoRes.imgToAnns[imgId]


    for k,v in res.items():
        if len(v)>1:
            res[k] = [v[-1]]

    # Tokenization
    eval_logger.info("tokenization...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    eval_logger.info(f"Computing {metric} scores...")

    score, scores = scorers_dict[metric][0].compute_score(gts, res)

    path = generate_submission_file(f"maxm_results_{metric}.json", args)
    if not os.path.exists(path):
        eval_logger.info("Storing prediction that can be submitted to the server ...")
        with open(path, "w") as f:
            json.dump(predictions, f, indent=4)

    return 100*score


def maxm_ema(results, args):
    return maxm_aggregate_results_v2(results, "exact_match", args)


def maxm_rouge_l(results, args):
    # print('Rouge input: ', results)
    return maxm_aggregate_results_v2(results, "rouge_l", args)

def maxm_cider(results, args):
    return maxm_aggregate_results_v2(results, "cider", args)


def maxm_relaxed_ema(results, args):
    return maxm_aggregate_results_v2(results, "relaxed_accuracy", args)