from simpletransformers.question_answering import QuestionAnsweringModel
from simpletransformers.language_modeling import LanguageModelingModel
from nested_lookup import nested_lookup
import pandas as pd
import numpy as np
import json
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
#nltk.download("punkt")
import string
import shutil
import datetime as dt
import torch
import sys
from glob import glob
import torch
import shutil
import os
from itertools import permutations


def fine_tuning(model_name, epoch, batch_size):

    model_base = model_name.split("/")[1].split("-")[0]
    train_args = {"reprocess_input_data": True,
                  "overwrite_output_dir": True,
                  }
    filename = "train-v0.1.json"
    with open('train_data.txt', 'w') as f:
        for item in data_list:
            f.write("%s\n" % item)
    # for some models seed may cause error
    try:
        lang_model = LanguageModelingModel(model_base, model_name, torch.Generator().manual_seed(42), use_cuda = True, args=train_args)
    except:
        lang_model = LanguageModelingModel(model_base, model_name, use_cuda = True, args=train_args)

    lang_output_file_name = "lang_output"
    lang_model.train_model("train_data.txt",output_dir = lang_output_file_name)
    necessary_files_for_pretrained_model = ['pytorch_model.bin', 'config.json', 'vocab.txt']
    lang_model_output_files = files = glob(lang_output_file_name + "/*" )
    files_to_be_removed = []
    for f in lang_model_output_files:
        if f.split("/")[1] not in necessary_files_for_pretrained_model:
          files_to_be_removed.append(f)
    for f in files_to_be_removed:
        if os.path.isfile(f):
            os.remove(f)
        elif os.path.isdir(f):
            shutil.rmtree(f)
    shutil.rmtree("cache_dir",ignore_errors=True)
    shutil.rmtree("runs",ignore_errors=True)

    train_args = {"reprocess_input_data": True,
                  "overwrite_output_dir": True,
                  "num_train_epochs" : epoch,
                  "train_batch_size" : batch_size
                  }

    qa_model = QuestionAnsweringModel(model_base, "lang_output", use_cuda = True,args = train_args)

    return qa_model

def soru_sor(soru,id, context,model):
    soru = [{
        'context': context,

        'qas': [
            {
                'question':soru,
                'id':id
            }
        ]
    }]

    return model.predict(soru)

def test_data_input(input_file):
    # Context into list
    f = open(input_file,"r",encoding = "utf-8")
    first_line = f.readline()
    del first_line
    # context
    context_list =    []
    context_line = True
    for line in f:
        if not line.strip():continue
        if line.startswith("context"):
            context_line = True
            a_list = []
        elif line.startswith("topic"):
            lines = ' '.join(a_list)
            context_list.append(lines)
            context_line = False
        if context_line:
            a_list.append(line.rstrip("\n").lstrip("\ufeff").strip("|").lstrip("context:").lstrip(": ").lstrip(" "))
    # questions into list
    f = open(input_file,"r",encoding = "utf-8")
    question_list =    []
    question_line = True
    for line in f:
        if not line.strip():continue
        if line.startswith("question"):
            question_line = True
            a = []
        elif line.startswith("answer"):
            lines = ' '.join(a)
            question_list.append(lines)
            question_line = False
        else:continue
        if question_line:
            a.append(line.rstrip("\n").lstrip("\ufeff").strip("|").lstrip("question:").lstrip(" ").rstrip(" "))
    # id into list
    f = open(input_file,"r",encoding = "utf-8")
    id_list =    []
    id_line = True
    for line in f:
        if not line.strip():continue
        if line.startswith("id"):
            id_line = True
        else:continue
        if id_line:
            id_list.append(line.rstrip("\n").lstrip("id:").strip("|").strip(" "))

    # To find number of questions for each topic
    f = open(input_file,"r",encoding = "utf-8")
    topic = []
    topic_line = True
    for line in f:
        if not line.strip():continue
        if line.startswith("topic"):
            topic_line = True
            a = []
        elif line.startswith("question"):
            lines = ' '.join(a)
            topic.append(lines)
            topic_line = False
        else:continue
        if topic_line:
            a.append(line.rstrip("\n").lstrip("topic:").rstrip("|").rstrip(" ").strip(" "))
    # answers
    f = open(input_file,"r",encoding = "utf-8")
    answer_list =    []
    answer_line = True
    for line in f:
        if not line.strip():continue
        if line.startswith("answer"):
            answer_line = True
            a = []
        elif line.startswith("id"):
            lines = ' '.join(a)
            answer_list.append(lines)
            answer_line = False
        else:continue
        if answer_line:
            a.append(line.rstrip("\n").strip("|").lstrip("answer:").lstrip(" ").rstrip(" "))
    # make contexts as much as multiplied by their questions
    topic_counter_dict = {}
    topic_counts = []
    for top in topic:
        if not top in topic_counter_dict.keys():
            topic_counter_dict[top] = 1
        elif top in topic_counter_dict.keys():
            topic_counter_dict[top] += 1
    for key,value in topic_counter_dict.items():
        topic_counts.append(value)
    a = [[con]*topic_counts[i] for i,con in enumerate(context_list)]
    context_list_with_num = []
    for context_list in a:
        for context_repeat in context_list:
            context_list_with_num.append(context_repeat)
    return context_list_with_num, question_list, id_list, topic, answer_list

def remove_stopwords(some_list):
    for token in some_list:
        if token in stopwords.words("turkish"):
            del some_list[some_list.index(token)]
    return some_list

def exact_match(predicted, actual):
    matched_answers = []
    guessed_true = 0
    for i, answer in enumerate(actual):
        if predicted[i].lower().translate(str.maketrans('', '', string.punctuation)).strip(" ") == actual[i].lower().translate(str.maketrans('', '', string.punctuation)).strip(" "):
            guessed_true += 1
            matched_answers.append(predicted[i])
    return guessed_true,matched_answers

### function for the whole process
def find_exact_match_number(model_idx, epoch_idx, batch_idx):

    t1 = dt.datetime.now()

    """
    istediginiz bir şey varsa (predictions, tokenized_preds gibi) return ettirebilirsiniz
    en başta cache_dir, outputs, runs klasörlerini sildiriyorum. çünkü bir model oluşturunca
    bunları da oluşturuyor ve bu klasörleri silmeden yeni model çağıramıyor
    %% capture her şeyi bastırmaması için (verbose = 0 gibi)
    """
    dirs = glob("*")
    dirs = pd.Series(dirs)
    dels = dirs[dirs.isin(["cache_dir","outputs","runs"])].to_list()

    for folder_name in dels:
        try:
            shutil.rmtree(folder_name)
        except:
            pass

    # this part for tuning: due to memory issues parameter combinations will be tried separately

    pretrained_models = ["dbmdz/bert-base-turkish-cased"
    ,"dbmdz/bert-base-turkish-128k-uncased"
    ,"savasy/bert-base-turkish-sentiment-cased"
    ,"dbmdz/bert-base-turkish-uncased"
    ,"savasy/bert-base-turkish-squad"
    ,"dbmdz/bert-base-turkish-128k-cased"
    ,"savasy/bert-turkish-text-classification"
    ,"dbmdz/electra-base-turkish-cased-discriminator"
    ,"savasy/bert-base-turkish-ner-cased"
    ,"lserinol/bert-turkish-question-answering"
    ,"dbmdz/distilbert-base-turkish-cased"
    ,"dbmdz/electra-small-turkish-cased-discriminator"
    ,"erayyildiz/electra-turkish-cased"
    ,"dbmdz/electra-base-turkish-cased-generator"
    ,"savasy/bert-turkish-uncased-qnli"
    ,"dbmdz/electra-base-turkish-cased-v0-discriminator"
    ,"dbmdz/electra-small-turkish-cased-generator"
    ,"dbmdz/electra-base-turkish-cased-v0-generator"]

    num_train_epochs = [1,2,3]
    train_batch_size = [16,32,128]

    model = fine_tuning(pretrained_models[model_idx], num_train_epochs[epoch_idx], train_batch_size[batch_idx])

    model.train_model(data_list,output_dir = "outputs")

    context_list_with_num, question_list, id_list, topic, answer_list = test_data_input("test_questions_compiled.txt")

    predicted_answers = []

    for i in range(len(context_list_with_num)):
        answerNprobs = (soru_sor(question_list[i], id_list[i], context_list_with_num[i], model))
        answer = answerNprobs[0][0]["answer"][0]
        predicted_answers.append(answer)

    tokenized_preds = [word_tokenize(sentence) for sentence in predicted_answers]
    tokenized_answers = [word_tokenize(sentence) for sentence in answer_list]
    predictions = [" ".join(remove_stopwords(token_list)) for token_list in tokenized_preds]
    actual_answers = [" ".join(remove_stopwords(token_list)) for token_list in tokenized_answers]
    exact_match_number, exact_match_answers = exact_match(predictions, actual_answers)

    t2 = dt.datetime.now()

    torch.cuda.empty_cache()

    return exact_match_number, t1, t2, predicted_answers, answer_list

def print_results(model_name, exact_match_number, t1, t2):

    print(model_name)
    print("Exact_match_number:{}".format(exact_match_number))
    print("Training + prediction took: " + str(t2-t1).split(".")[0] + " (format: hh:mm:ss)")
    print(" ")

# action starts here

model_idx = int(sys.argv[1])
epoch_idx = int(sys.argv[2])
batch_idx = int(sys.argv[3])

with open('train-v0.1.json' , encoding='utf-8') as json_file:
    data = json.load(json_file)

data_list = list()
for i in data['data']:
    for j in i['paragraphs']:
        for k in j['qas']:
            k['is_impossible'] = False
            for z in k['answers']:
                tmp = z['answer_start']
                z['answer_start'] = int(tmp)
        data_list.append(j)

exact_match_number, t1, t2, predicted_answers, answer_list = find_exact_match_number(model_idx, epoch_idx, batch_idx)


pretrained_models = ["dbmdz/bert-base-turkish-cased"
,"dbmdz/bert-base-turkish-128k-uncased"
,"savasy/bert-base-turkish-sentiment-cased"
,"dbmdz/bert-base-turkish-uncased"
,"savasy/bert-base-turkish-squad"
,"dbmdz/bert-base-turkish-128k-cased"
,"savasy/bert-turkish-text-classification"
,"dbmdz/electra-base-turkish-cased-discriminator"
,"savasy/bert-base-turkish-ner-cased"
,"lserinol/bert-turkish-question-answering"
,"dbmdz/distilbert-base-turkish-cased"
,"dbmdz/electra-small-turkish-cased-discriminator"
,"erayyildiz/electra-turkish-cased"
,"dbmdz/electra-base-turkish-cased-generator"
,"savasy/bert-turkish-uncased-qnli"
,"dbmdz/electra-base-turkish-cased-v0-discriminator"
,"dbmdz/electra-small-turkish-cased-generator"
,"dbmdz/electra-base-turkish-cased-v0-generator"]

num_train_epochs = [1,2,3]
train_batch_size = [16,32,128]


res = pd.DataFrame(predicted_answers + [pretrained_models[model_idx]] + [num_train_epochs[epoch_idx]] + [train_batch_size[batch_idx]], columns = ["predicted"])
res["true"] = answer_list + [str(t2-t1).split(".")[0]] + ["",""]
csv_path = "fine_tuning_results/" + (pretrained_models[model_idx] + "_epoch_" + str(num_train_epochs[epoch_idx]) + "_batchsize_" + str(train_batch_size[batch_idx]) +   ".csv").replace("/", "_")
res.to_csv(csv_path, index = None)
