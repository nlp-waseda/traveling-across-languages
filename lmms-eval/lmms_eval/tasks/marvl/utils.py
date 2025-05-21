
def marvl_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def marvl_doc_to_text(doc, model_specific_prompt_kwargs):
    conversations = doc['conversations']
    query = conversations[0]['value'].replace('<image>\n', '').strip() + '. Answer with the option directly.'
    return query
    
def marvl_doc_to_target(doc):
    conversations = doc['conversations']
    answer = str(conversations[1])
    if 'true' in answer: return 'true'
    elif 'false' in answer: return 'false'
    else: raise Exception(f"get target failed for id {doc['id']} - conversations: {conversations}")

def marvl_process_result(doc, results):
    target = marvl_doc_to_target(doc)
    pred = results[0]
    pred = pred.strip().lower()
    if 'yes' in pred: pred = 'true'
    elif 'no' in pred: pred = 'false'
    if target.strip().lower() in pred:
        return {"exact_match": 1.0}
    return {"exact_match": 0.0}
