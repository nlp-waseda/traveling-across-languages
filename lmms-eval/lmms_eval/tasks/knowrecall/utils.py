from pathlib import Path

from jinja2 import Template


def knowrecall_doc_to_text(doc):
    question = doc["question"]
    choices = doc["options"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join(
        [f"{option}. {choice}" for option, choice in zip(options, choices)]
    )

    prompt_path = Path(f"{Path(__file__).resolve().parent}/prompt.j2")
    prompt_template = Template(prompt_path.read_text(encoding="utf-8"))
    return prompt_template.render(question=question, choices_str=choices_str)


def knowrecall_doc_to_visual(doc):
    if doc["image"] is None:
        return []
    return [doc["image"].convert("RGB")]


def knowrecall_doc_to_target(doc):
    choices = doc["options"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    answer_index = choices.index(doc["answer"])
    return options[answer_index]


def knowrecall_doc_to_custom_str(doc):
    return doc["domestic_language_code"]
