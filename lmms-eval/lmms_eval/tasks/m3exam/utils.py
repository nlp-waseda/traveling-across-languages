import re
import logging
import random
from difflib import SequenceMatcher
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
import io
import base64
lmms_logger = logging.getLogger("lmms-eval")

LANG_CONFIG = {
    'english': "Answer:",
    'chinese': '答案：',
    'vietnamese': 'Câu trả lời:',
    'thai': 'คำตอบ:',
    'italian': 'La risposta:',
    'afrikaans': 'Antwoord:',
    'portuguese': 'Responder:'
}
    
def construct_prompt(doc):
    lang = doc["language"]
    subject2target = {
        'english': {'language': 'English', 'math': "Math", 'social-science': "Social Science", 'natural-science': 'Natural Science'},
        # 'english4all': {'language': 'Language', 'math': "Math", 'social-science': "Social Science", 'natural-science': 'Natural Science'},
        'chinese':  {'language': '语文', 'math': "数学", 'social-science': "社会科学", 'natural-science': '自然科学'},
        'thai': {'language': 'ภาษาไทย', 'math': 'คณิตศาสตร์', 'social-science': 'สังคมศึกษา', 'natural-science': 'วิทยาศาสตร์'},
        'vietnamese': {'language': 'Tiếng Việt', 'math': "Toán", 'social-science': "Khoa học xã hội", 'natural-science': 'Khoa học tự nhiên'},
        'italian': {'language': 'Italiano', 'math': "Matematica", 'social-science': "Scienze sociali", 'natural-science': 'Scienze naturali'},
        'afrikaans': {'language': 'Afrikaans Huistaal', 'math': "Wiskunde", 'social-science': "Sosiale Wetenskappe", 'natural-science': 'Natuurwetenskap'},
        'portuguese': {'language': 'Linguagens', 'math': 'Matemática', 'social-science': 'Ciências Humanas', 'natural-science': 'Ciências da Natureza'},
    }
    subject = subject2target[lang][doc['subject_category']]

    hint_templates = {
        'english': f"The following is a multiple choice question about {subject}. Please only respond with the letter (A, B, C, or D) corresponding to the correct answer, without any additional explanation.",
        'chinese': f"以下是关于{subject}的单项选择题。 请仅给出正确选项对应的选项序号（A, B, C, 或 D） 而非其他细节。",
        'thai': f"ต่อไปนี้เป็นคำถามแบบปรนัย วิชา{subject} โปรดตอบเพียงตัวอักษร (A, B, C หรือ D) ที่ตรงกับคำตอบที่ถูกต้อง โดยไม่ต้องมีคำอธิบายเพิ่มเติม",
        'vietnamese': f"Sau đây là câu hỏi trắc nghiệm về {subject}. Vui lòng chỉ trả lời bằng chữ cái (A, B, C hoặc D) tương ứng với câu trả lời đúng, không cần giải thích thêm.",
        'italian': f"La seguente è una domanda a scelta multipla su {subject}. Si prega di rispondere solo con la lettera (A, B, C o D) corrispondente alla risposta corretta, senza alcuna spiegazione aggiuntiva.",
        'afrikaans': f"Die volgende is 'n meervoudige keuse vraag oor {subject}. Antwoord asseblief slegs met die letter (A, B, C of D) wat ooreenstem met die korrekte antwoord, sonder enige bykomende verduideliking.",
        'portuguese': f"A seguir está uma questão de múltipla escolha sobre {subject}. Por favor, responda apenas com a letra (A, B, C ou D) correspondente à resposta correta, sem qualquer explicação adicional."
    }

    hint = hint_templates.get(lang, "")
    prompt = f"{hint}\n\n<prompt_placeholder>"
    return prompt

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def parse_multi_choice_response(response, options):
    response = response.strip()
    
    # Original letter-matching logic
    match = re.search(r'\(?([A-D])[).:\s]', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # If no match found, fallback to searching for any A, B, C, or D in the response
    match = re.search(r'[ABCD]', response, re.IGNORECASE)
    if match:
        return match.group(0).upper()

    # If no letter found, match full content
    best_match = None
    best_match_ratio = 0
    for i, option in enumerate(options):
        option_content = re.sub(r'^[A-D]\.\s*', '', option).strip()
        similarity = similar(response, option_content)
        if similarity > best_match_ratio:
            best_match = chr(65 + i)  # 'A', 'B', 'C', or 'D'
            best_match_ratio = similarity

    # If we found a good match (you can adjust the threshold)
    if best_match_ratio > 0.7:
        return best_match

    # If all else fails, return a random choice
    return random.choice(['A', 'B', 'C', 'D'])

def m3exam_process_results(doc, results):
    pred = results[0]
    parsed_pred = parse_multi_choice_response(pred,doc['options'])
    standardized_answer = standardize_answer(doc["answer_text"], doc['options'])

    return {
        "m3exam": {
            "language": doc["language"],
            "origin_response": pred,
            "answer_text": parsed_pred,
            "origin_answer": doc["answer_text"],
            "standardized_answer": standardized_answer
        }
    }

def m3exam_aggregate_results(results):
    total, match = 0, 0
    for question in results:
        total += 1
        if question["answer_text"] == question["standardized_answer"]:
            match += 1
    
    accuracy = match / total if total > 0 else 0
    print(f"==========================")
    print(f"========Final Score=======")
    print(f"Total questions: {total}")
    print(f"Correct answers: {match}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"==========================")
    
    return accuracy


def replace_images_tokens(input_string, image_ids):
    if len(image_ids) == 0:
        return input_string
    else:
        for id in image_ids:
            image_str = f"(image)[{id}]"
            query_text = "<image>"
            if image_str in input_string:
                input_string = input_string.replace(image_str, query_text)
    return input_string

def standardize_options(options):
    standardized = []
    for i, option in enumerate(options):
        # Remove any existing option identifier
        cleaned = re.sub(r'^[\(（]?[A-Da-d\d][\)）\.\s]+\s*', '', option).strip()
        # Add standardized identifier
        standardized.append(f"({chr(65+i)}) {cleaned}")
    return standardized


def standardize_answer(answer, options):
    # If answer is already A, B, C, or D, return it
    if answer in 'ABCDEFGH':
        return answer
    
    # If answer is 1, 2, 3, or 4, convert to A, B, C, or D
    if answer in '12345678':
        return chr(64 + int(answer))
    
    # If answer is (1), (2), (3), or (4), convert to A, B, C, or D
    match = re.match(r'\(?(\d)\)?', answer)
    if match:
        return chr(64 + int(match.group(1)))
    
    # If we can't determine the answer, return the original
    return answer

def get_text_dimensions(text, font):
    # This function handles both newer and older Pillow versions
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        return font.getsize(text)

def combine_images_with_labels(images, labels):
    """
    Combine multiple images into a single image, adding labels with dynamically sized font.
    """
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)
    padding = 20  # Increased padding for larger labels

    combined_img = PILImage.new('RGB', (max_width, total_height + padding * (len(images) - 1)), color='white')
    draw = ImageDraw.Draw(combined_img)
    
    # Calculate dynamic font size
    base_font_size = int(min(max_width, total_height) * 0.03)  # 3% of the smaller dimension
    font_size = max(24, min(base_font_size, 72))  # Minimum 24, maximum 72
    
    try:
        font = ImageFont.truetype("OpenSans-Regular.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    y_offset = 0
    for i, img in enumerate(images):
        combined_img.paste(img, (0, y_offset))
        label = labels[i]
        
        # Calculate text position
        text_width, text_height = get_text_dimensions(label, font)
        text_x = 0  # 10 pixels from the left edge
        text_y = y_offset - 10  # 10 pixels from the top of each image
        
        # Add a semi-transparent background for the text for better readability
        text_bg = PILImage.new('RGBA', (text_width + 20, text_height + 20), (255, 255, 255, 128))
        combined_img.paste(text_bg, (text_x - 10, text_y - 10), text_bg)
        
        # Draw the text
        draw.text((text_x, text_y), label, fill="black", font=font)
        
        y_offset += img.height + padding

    return combined_img

def generate_white_image(width=300, height=200):
    """Generate a white image of the specified size."""
    return PILImage.new('RGB', (width, height), color='white')

def m3exam_doc_to_visual(doc):
    visual = []
    for i in range(10):
        if doc['image_' + str(i)] == "None":
            continue
        else:
            image_data = base64.b64decode(doc['image_' + str(i)])
            image = PILImage.open(io.BytesIO(image_data))
    
            image_rgb = image.convert('RGB')

            visual.append(image_rgb)

    return visual
    
def m3exam_doc_to_text(doc):
    # Standardize options
    doc['options'] = standardize_options(doc['options'])
    lang = doc["language"]

    # Process question text
    question_text = re.sub(r'\(image\)\[[^\]]+\]', '<image>', doc['question_text'], count=1)
    question_text = re.sub(r'\(image\)\[[^\]]+\]', '', question_text)
    
    # Prepare options text
    if len(re.findall(r'\(image\)\[[^\]]+\]', ' '.join(doc['options']))) > 0:
        # If options contain images, don't include option text
        options_text = ''
    else:
        options_text = '\n'.join(doc['options'])
    
    # Construct prompt
    prompt = construct_prompt(doc)
    background = '\n' + '\n'.join(doc['background_description']) if doc['background_description'] else ''
    prompt = prompt.replace("<prompt_placeholder>", f"{background}\n{question_text}\n{options_text}\n{LANG_CONFIG[lang]}")

    return prompt