
def xgqa_doc_to_visual(doc):
    return [doc['image'].convert("RGB")]


def xgqa_doc_to_text(doc, model_specific_prompt_kwargs):
    question = doc["question"].strip()
    pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    post_prompt = model_specific_prompt_kwargs["post_prompt"]
    return f"{pre_prompt}{question} {post_prompt.strip()}"

def xgqa_process_result(doc, results):
    target = doc['answer'].strip().lower()
    pred = results[0]
    pred = pred.strip().lower()
    if target in pred:
        return {"exact_match": 1.0}
    return {"exact_match": 0.0}