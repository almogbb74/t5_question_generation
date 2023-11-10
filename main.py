import requests
import nltk
import torch
from random import shuffle
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk import tokenize
from nltk.corpus import stopwords
from operator import itemgetter
import math
from flask import Flask, request

app = Flask(__name__)


def load_wikipedia_json(value):  # Get info about a subject from Wikipedia
    wikipedia_json_url = f'https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro' \
                         f'&explaintext&redirects=1&titles={value.replace(" ", "_")}'
    json_data = requests.get(wikipedia_json_url).json()
    page_id = str(json_data['query']['pages']).split(':')[0].removeprefix('{\'').removesuffix('\'')
    if page_id != '-1':
        return json_data['query']['pages'][page_id]['extract']
    return -1


def calc_tf(info, stop_word_tf):
    # Term frequency - frequency of each term in a sentence
    total_words = info.split()
    total_sentences = tokenize.sent_tokenize(info)
    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_word_tf:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1

    # Dividing by the total number of words for each dictionary element
    tf_score.update((x, (y / int(len(total_words)))) for x, y in tf_score.items())
    return tf_score, total_words, total_sentences


def word_in_sent(word, sentences):  # Find in how many sentences the term is in
    final = [all([w in x for w in word]) for x in sentences]  # final[i] will be True if word is in sentence i
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))


def calc_idf(total_words, stop_words_idf, total_sentences):
    # Inverse document frequency - log ([number of sentences that the term is in] / [total number of sentences])
    idf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.', '')
        if each_word not in stop_words_idf:
            if each_word in idf_score:
                idf_score[each_word] = word_in_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1

    # Performing a log and divide
    idf_score.update((x, math.log(int(len(total_sentences)) / y)) for x, y in idf_score.items())
    return idf_score


def calc_tf_idf(tf_result, idf_result):  # Calculate tf * idf
    tf_idf_score = {key: tf_result[key] * idf_result.get(key, 0) for key in tf_result.keys()}
    return tf_idf_score


def get_top_n(dict_elem, n):  # Get top n keywords
    result = dict(sorted(dict_elem.items(), key=itemgetter(1), reverse=True)[:n])
    return result


def find_keywords(subject):
    data = load_wikipedia_json(subject)  # Get info from wikipedia
    if data != -1:
        try:  # Check if packages are installed
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')
        stop_words = set(stopwords.words('english'))
        func_result = calc_tf(data, stop_words)
        # Tuple contains tf score for each word, total words, total sentences, and total sentences
        tf_score_result = func_result[0]
        total_word_result = func_result[1]
        total_sentences_result = func_result[2]
        idf_score_result = calc_idf(total_word_result, stop_words,
                                    total_sentences_result)  # Calculate idf (inverse document frequency)
        tf_idf_result = calc_tf_idf(tf_score_result, idf_score_result)
        return data, get_top_n(tf_idf_result, 3)
    return -1, -1


def load_questions(subject):
    trained_model_path = 't5/model/'
    trained_tokenizer = 't5/tokenizer/'
    context, answer = find_keywords(subject)
    if context != -1:
        model = T5ForConditionalGeneration.from_pretrained(trained_model_path)
        tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer)
        device = torch.device("cpu")
        print(f'\ndevice {device} \n')
        model = model.to(device)
        questions_list = []
        print("Model is generating questions...\n")
        for i in range(3):
            text = "context: " + context + " " + "answer: " + list(answer.keys())[i]
            encoding = tokenizer.encode_plus(text, max_length=512, padding='max_length', return_tensors="pt")
            input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
            # input_ids- a list with tokens, each of the numbers (each token is a number) in the input_ids list refers
            # to a token ID which has a corresponding string in the T5 vocabulary.

            # attention_mask - a list with 0s and 1s. puts if input_ids[i] != 0. in other words, a mask of input_ids.
            model.eval()
            beam_outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_length=72,
                early_stopping=True,
                num_beams=5,
                num_return_sequences=3)

            for beam_output in beam_outputs:
                potential_question = tokenizer.decode(beam_output, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=True)
                if potential_question not in questions_list:
                    print("Question found!")
                    questions_list.append(potential_question.replace("question: ", ""))
        shuffle(questions_list)
        return questions_list

    else:
        return 'Could not find information about subject'


@app.route("/")
def send_questions_to_app():
    subject = request.args.get("subject")
    questions_list = load_questions(subject)
    if questions_list != 'Could not find information about subject':
        print(questions_list, sep='\n')
        questions_string = ",".join(questions_list)
        return questions_string
    print('Could not find information about subject')
    return '-1'


if __name__ == "__main__":
    app.run(host="0.0.0.0")
