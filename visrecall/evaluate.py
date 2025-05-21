import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel

sys.path.append(str(Path("..").resolve()))
from utils.country_language_code import language_codes_des as language_codes

TRUNCATE_DIM=512
W=2.5


def main():
    current_path = Path(__file__).parent.resolve()

    # read model predictions
    with open(args.prediction_file) as f:
        predictions = json.load(f)
        predictions = dict(sorted(predictions.items(), key=lambda x: x[0]))

    # read language info
    with open(current_path / Path("landmark_list.json")) as f:
        landmarks = json.load(f)
        landmark_id_to_domestic_lan = {}
        for landmark in landmarks:
            landmark_id_to_domestic_lan[str(landmark["landmark_id"])] = \
                landmark["domestic_language_code"]

    # read image list
    with open(current_path / Path("image_list.json")) as f:
        image_list = json.load(f)
        image_list = dict(sorted(image_list.items(), key=lambda x: x[0]))

    # load CLIP model
    model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
    model = model.cuda()
    model.eval()

    # calculate CLIPScore
    results = {}
    for landmark_id in tqdm(predictions):
        results[landmark_id] = {}

        prediction = predictions[landmark_id]
        images = image_list[landmark_id]
        images = [os.path.join(current_path / Path("images"), landmark_id, image) for image in images]

        # encode images
        image_embeddings = model.encode_image(images, truncate_dim=TRUNCATE_DIM)
        image_embeddings = image_embeddings / np.sqrt(
            np.sum(image_embeddings**2, axis=1, keepdims=True),
        )

        for lan in prediction:
            prediction_lan = prediction[lan]
            if "Error" in prediction_lan:
                prediction_lan = [x for x in prediction_lan if x != "Error"]
            if len(prediction_lan) == 0:
                results[landmark_id][lan] = {
                    "avg": 0,
                    "per_prompt": [0, 0],
                }
                continue

            # encode text
            text_embeddings = model.encode_text(prediction_lan, truncate_dim=TRUNCATE_DIM)
            text_embeddings = text_embeddings / np.sqrt(
                np.sum(text_embeddings**2, axis=1, keepdims=True),
            )
            # calculate scores
            scores_matrix = np.dot(image_embeddings, text_embeddings.T)
            weighted_scores = W * np.clip(scores_matrix, 0, None)
            results[landmark_id][lan] = {
                "avg": float(np.mean(weighted_scores)),
                "per_prompt": [float(x) for x in np.mean(weighted_scores, axis=0)],
            }

    # aggregate scores
    agg_results = {}
    agg_results["all"] = {}

    # mean of per language
    for lan in language_codes:
        scores = [results[landmark_id][lan]["avg"] for landmark_id in results]
        agg_results["all"][lan] = float(np.mean(scores))

    # mean of domestic language
    scores = [
        results[landmark_id][landmark_id_to_domestic_lan[landmark_id]]["avg"]
        for landmark_id in results
    ]
    agg_results["all"]["domestic"] = float(np.mean(scores))

    # mean of inbound languages
    scores = [
        results[landmark_id][lan]["avg"]
        for landmark_id in results
        for lan in language_codes
        if lan != "en" and lan != landmark_id_to_domestic_lan.get(landmark_id)
    ]
    agg_results["all"]["inbound"] = float(np.mean(scores))

    # aggregate scores for each language region
    for lan in language_codes:
        agg_results[lan] = {}
        for lan_ in language_codes:
            scores = [
                results[landmark_id][lan_]["avg"]
                for landmark_id in results
                if lan == landmark_id_to_domestic_lan[landmark_id]
            ]
            agg_results[lan][lan_] = float(np.mean(scores))
        domestic_scores = [
            results[landmark_id][lan]["avg"]
            for landmark_id in results
            if lan == landmark_id_to_domestic_lan[landmark_id]
        ]
        agg_results[lan]["domestic"] = float(np.mean(domestic_scores))
        inbound_scores = [
            results[landmark_id][lan_]["avg"]
            for landmark_id in results
            for lan_ in language_codes
            if lan == landmark_id_to_domestic_lan[landmark_id] and lan_  not in ["en", lan]
        ]
        agg_results[lan]["inbound"] = float(np.mean(inbound_scores))

    results["agg"] = agg_results

    # calculate consistency matrix
    consis_matrix = {}
    for lan_x in language_codes:
        consis_matrix[lan_x] = {}

        lan_x_preds = []
        for landmark_id in (key for key in results if key != "agg"):
            if landmark_id_to_domestic_lan.get(landmark_id) != lan_x:
                continue
            lan_x_preds.append(results[landmark_id][lan_x]["avg"])

        for lan_y in language_codes:
            lan_y_preds = []
            for landmark_id in (key for key in results if key != "agg"):
                if landmark_id_to_domestic_lan.get(landmark_id) != lan_x:
                    continue
                lan_y_preds.append(results[landmark_id][lan_y]["avg"])

            n_x, n_y, n_xy = sum(lan_x_preds), sum(lan_y_preds), 0
            for lan_x_pred, lan_y_pred in zip(lan_x_preds, lan_y_preds, strict=True):
                n_xy += min(lan_x_pred, lan_y_pred)
            consis_matrix[lan_x][lan_y] = ((n_xy / n_y) + (n_xy / n_x)) / 2

    results["mean_consistency"] = pd.DataFrame(consis_matrix).fillna(0).mean().mean()

    base_name = os.path.basename(args.prediction_file)
    os.makedirs(current_path / Path("evaluation_results"), exist_ok=True)
    with open(current_path / Path(f"evaluation_results/{base_name}"), "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str, required=True)
    args = parser.parse_args()
    main()
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel

sys.path.append(str(Path("..").resolve()))
from utils.country_language_code import language_codes_des as language_codes

TRUNCATE_DIM=512
W=2.5


def main():
    current_path = Path(__file__).parent.resolve()

    # read model predictions
    with open(args.prediction_file) as f:
        predictions = json.load(f)
        predictions = dict(sorted(predictions.items(), key=lambda x: x[0]))

    # read language info
    with open(current_path / Path("landmark_list.json")) as f:
        landmarks = json.load(f)
        landmark_id_to_domestic_lan = {}
        for landmark in landmarks:
            landmark_id_to_domestic_lan[str(landmark["landmark_id"])] = \
                landmark["domestic_language_code"]

    # read image list
    with open(current_path / Path("image_list.json")) as f:
        image_list = json.load(f)
        image_list = dict(sorted(image_list.items(), key=lambda x: x[0]))

    # load CLIP model
    model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
    model = model.cuda()
    model.eval()

    # calculate CLIPScore
    results = {}
    for landmark_id in tqdm(predictions):
        results[landmark_id] = {}

        prediction = predictions[landmark_id]
        images = image_list[landmark_id]
        images = [os.path.join(current_path / Path("images"), landmark_id, image) for image in images]

        # encode images
        image_embeddings = model.encode_image(images, truncate_dim=TRUNCATE_DIM)
        image_embeddings = image_embeddings / np.sqrt(
            np.sum(image_embeddings**2, axis=1, keepdims=True),
        )

        for lan in prediction:
            prediction_lan = prediction[lan]
            if "Error" in prediction_lan:
                prediction_lan = [x for x in prediction_lan if x != "Error"]
            if len(prediction_lan) == 0:
                results[landmark_id][lan] = {
                    "avg": 0,
                    "per_prompt": [0, 0],
                }
                continue

            # encode text
            text_embeddings = model.encode_text(prediction_lan, truncate_dim=TRUNCATE_DIM)
            text_embeddings = text_embeddings / np.sqrt(
                np.sum(text_embeddings**2, axis=1, keepdims=True),
            )
            # calculate scores
            scores_matrix = np.dot(image_embeddings, text_embeddings.T)
            weighted_scores = W * np.clip(scores_matrix, 0, None)
            results[landmark_id][lan] = {
                "avg": float(np.mean(weighted_scores)),
                "per_prompt": [float(x) for x in np.mean(weighted_scores, axis=0)],
            }

    # aggregate scores
    agg_results = {}
    agg_results["all"] = {}

    # mean of per language
    for lan in language_codes:
        scores = [results[landmark_id][lan]["avg"] for landmark_id in results]
        agg_results["all"][lan] = float(np.mean(scores))

    # mean of domestic language
    scores = [
        results[landmark_id][landmark_id_to_domestic_lan[landmark_id]]["avg"]
        for landmark_id in results
    ]
    agg_results["all"]["domestic"] = float(np.mean(scores))

    # mean of inbound languages
    scores = [
        results[landmark_id][lan]["avg"]
        for landmark_id in results
        for lan in language_codes
        if lan != "en" and lan != landmark_id_to_domestic_lan.get(landmark_id)
    ]
    agg_results["all"]["inbound"] = float(np.mean(scores))

    # aggregate scores for each language region
    for lan in language_codes:
        agg_results[lan] = {}
        for lan_ in language_codes:
            scores = [
                results[landmark_id][lan_]["avg"]
                for landmark_id in results
                if lan == landmark_id_to_domestic_lan[landmark_id]
            ]
            agg_results[lan][lan_] = float(np.mean(scores))
        domestic_scores = [
            results[landmark_id][lan]["avg"]
            for landmark_id in results
            if lan == landmark_id_to_domestic_lan[landmark_id]
        ]
        agg_results[lan]["domestic"] = float(np.mean(domestic_scores))
        inbound_scores = [
            results[landmark_id][lan_]["avg"]
            for landmark_id in results
            for lan_ in language_codes
            if lan == landmark_id_to_domestic_lan[landmark_id] and lan_  not in ["en", lan]
        ]
        agg_results[lan]["inbound"] = float(np.mean(inbound_scores))

    results["agg"] = agg_results

    # calculate consistency matrix
    consis_matrix = {}
    for lan_x in language_codes:
        consis_matrix[lan_x] = {}

        lan_x_preds = []
        for landmark_id in (key for key in results if key != "agg"):
            if landmark_id_to_domestic_lan.get(landmark_id) != lan_x:
                continue
            lan_x_preds.append(results[landmark_id][lan_x]["avg"])

        for lan_y in language_codes:
            lan_y_preds = []
            for landmark_id in (key for key in results if key != "agg"):
                if landmark_id_to_domestic_lan.get(landmark_id) != lan_x:
                    continue
                lan_y_preds.append(results[landmark_id][lan_y]["avg"])

            n_x, n_y, n_xy = sum(lan_x_preds), sum(lan_y_preds), 0
            for lan_x_pred, lan_y_pred in zip(lan_x_preds, lan_y_preds, strict=True):
                n_xy += min(lan_x_pred, lan_y_pred)
            consis_matrix[lan_x][lan_y] = ((n_xy / n_y) + (n_xy / n_x)) / 2

    results["mean_consistency"] = pd.DataFrame(consis_matrix).fillna(0).mean().mean()

    base_name = os.path.basename(args.prediction_file)
    os.makedirs(current_path / Path("evaluation_results"), exist_ok=True)
    with open(current_path / Path(f"evaluation_results/{base_name}"), "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str, required=True)
    args = parser.parse_args()
    main()
