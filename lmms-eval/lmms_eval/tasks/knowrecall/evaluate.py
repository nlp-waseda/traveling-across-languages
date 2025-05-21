import argparse
import json
import os
import re
from difflib import SequenceMatcher

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

LANGUAGE_CODES = ["ar", "de", "el", "en", "es", "fr", "he", "it", "ja", "ko", "pt", "ru", "sr", "th", "zh"]


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def parse_multi_choice_response(response: str, options: list[str]) -> str:
    response = response.strip()

    # Direct single letter answer
    if response in ["A", "B", "C", "D"]:
        return response

    # Try to extract answer from JSON block or inline JSON
    match = re.search(r"```json\s*({.*?})\s*```", response, re.DOTALL) or re.search(r"({.*?})", response, re.DOTALL)
    if match:
        try:
            response = json.loads(match.group(1)).get("answer")
            if response is None:
                return "Error: No answer found"
            return response
        except Exception:
            pass

    # Try to match response to options by similarity
    best, score = None, 0
    for i, option in enumerate(options):
        option_text = re.sub(r"^[A-D]\.\s*", "", option).strip()
        sim = similar(response, option_text)
        if sim > score:
            best, score = chr(65 + i), sim
    if score > 0.7:
        return best

    # Try to extract answer letter from text
    match = re.search(r"\(?([A-D])[).:\s]", response)
    if match:
        return match.group(1).upper()

    return "Error: No answer found"


def read_predictions(log_dir: str) -> tuple[dict, dict, dict]:
    results = {lan: {lan_: 0 for lan_ in LANGUAGE_CODES} for lan in LANGUAGE_CODES}
    labels = {lan: {lan_: [] for lan_ in LANGUAGE_CODES} for lan in LANGUAGE_CODES}
    preds = {lan: {lan_: [] for lan_ in LANGUAGE_CODES} for lan in LANGUAGE_CODES}
    lanmark_id_to_domestic_lan = {}

    for language in LANGUAGE_CODES:
        predictions = None
        candidate_patterns = [
            f"knowrecall_{language}.json",
            f"knowrecall_{language}_zeroshot_cot.json",
            f"knowrecall_{language}_structured_cot.json",
        ]
        for pattern in candidate_patterns:
            candidate_path = os.path.join(log_dir, pattern)
            if os.path.isfile(candidate_path):
                with open(candidate_path) as f:
                    predictions = json.load(f)
                break

        for item in predictions["logs"]:
            doc = item["doc"]
            domestic_language_code = doc["domestic_language_code"]
            lanmark_id_to_domestic_lan[doc["landmark_id"]] = domestic_language_code
            pred = parse_multi_choice_response(item["resps"][0][0], doc["options"])
            answer = item["prediction"]["answer"]

            if pred == answer:
                results[domestic_language_code][language] += 1

            labels[domestic_language_code][language].append(answer)
            preds[domestic_language_code][language].append(pred)

    return results, labels, preds


def calculate_accuracy(results: dict) -> dict:
    for lan in LANGUAGE_CODES:
        for lan_ in LANGUAGE_CODES:
            results[lan][lan_] = results[lan][lan_] / 200

    for lan in LANGUAGE_CODES:
        results[lan]["domestic"] = results[lan][lan]
        results[lan]["inbound"] = sum(
            [results[lan][lan_] for lan_ in LANGUAGE_CODES if lan_ not in ["en", lan]]
        ) / len([lan_ for lan_ in LANGUAGE_CODES if lan_ not in ["en", lan]])

    results["all"] = {lan: sum([results[lan_][lan] for lan_ in LANGUAGE_CODES]) / len(LANGUAGE_CODES) for lan in LANGUAGE_CODES}
    results["all"]["domestic"] = sum([results[lan]["domestic"] for lan in LANGUAGE_CODES]) / len(LANGUAGE_CODES)
    scores = [results[lan][lan_] for lan in LANGUAGE_CODES for lan_ in LANGUAGE_CODES if lan_ not in ["en", lan]]
    results["all"]["inbound"] = sum(scores) / len(scores)

    return results


def calculate_consistency_matrix(labels: dict, preds: dict) -> dict:
    consis_matrix = {}
    for x in LANGUAGE_CODES:
        consis_matrix[x] = {}
        x_labels, x_preds = labels[x][x], preds[x][x]
        n_x = sum(x_pred == label for x_pred, label in zip(x_preds, x_labels))

        for y in LANGUAGE_CODES:
            _, y_preds = labels[x][y], preds[x][y]
            n_y = sum(y_pred == label for y_pred, label in zip(y_preds, x_labels))
            n_xy = sum(x_pred == y_pred == label for x_pred, y_pred, label in zip(x_preds, y_preds, x_labels))
            consis_matrix[x][y] = ((n_xy / n_y) + (n_xy / n_x)) / 2 if n_x and n_y else 0
    return consis_matrix


def main():
    parser = argparse.ArgumentParser(description="Evaluate KnowRecall predictions.")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing the log files.")
    args = parser.parse_args()

    results, labels, preds = read_predictions(args.log_dir)
    evaluation_results = calculate_accuracy(results)

    consis_matrix = calculate_consistency_matrix(labels, preds)
    consis_matrix_df = pd.DataFrame(consis_matrix).fillna(0)
    consis_matrix_df = consis_matrix_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(consis_matrix_df, cmap="coolwarm", center=0.75, annot=True, fmt=".2f", ax=ax, vmin=0.6, vmax=1)
    plt.xlabel("Language 1")
    plt.ylabel("Language 2")
    plt.title("Consistency")
    plt.savefig(os.path.join(args.log_dir, "consistency.png"), dpi=300)

    mean_consistency = consis_matrix_df.mean().mean()
    evaluation_results["mean_consistency"] = mean_consistency

    with open(os.path.join(args.log_dir, "evaluation_results.json"), "w") as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
