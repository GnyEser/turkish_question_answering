import pandas as pd
import numpy as np
import collections
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download("stopwords")
nltk.download("punkt")
import os
import re
# Extended Match
stopwords_list = set(stopwords.words("turkish"))
stopwords_list.update(("kadar", "dır","dir","yaklaşık","sidir","sıdır","den","dan","kat","tan","ten","dan","den","ile"))
def extended_match(prediction,actual):
  # tokenize
  # predictions
  tokenized_preds = [word_tokenize(sentence) for sentence in prediction]
  # actual answers
  tokenized_answers = [word_tokenize(sentence) for sentence in actual]
  # remove stopwords function
  def remove_stopwords(some_list):
    for token in some_list:
      if token in stopwords_list:
        del some_list[some_list.index(token)]
    return some_list
  # remove stopwords from predictions and answers
  predictions = [" ".join(remove_stopwords(token_list)) for token_list in tokenized_preds]
  actual_answers = [" ".join(remove_stopwords(token_list)) for token_list in tokenized_answers]
  # remove punctuations,lower text from predictions and answers
  predictions = [prediction.lower().translate(str.maketrans('', '', string.punctuation)).strip(" ") for prediction in predictions]
  actual_answers = [answer.lower().translate(str.maketrans('', '', string.punctuation)).strip(" ") for answer in actual_answers]
  # get a list of unmatched predictions
  unmatched_predictions = []
  # get a list of unmatched answers
  unmatched_answers = []
  # get a list of matched predictions
  matched_preds = []
  # get a list of matched answers
  matched_answers = []
  # count of true guesses
  guessed_true = 0
  for i, answer in enumerate(actual_answers):
    if (predictions[i] == actual_answers[i])|((predictions[i] in actual_answers[i])&(len(predictions[i])!=0)):
      guessed_true += 1 
      matched_preds.append(predictions[i])
      matched_answers.append(actual_answers[i])
    elif (predictions[i] != actual_answers[i])&(len(predictions[i])!=0):
      unmatched_answers.append(actual_answers[i])
      unmatched_predictions.append(predictions[i])
  return round(guessed_true/len(prediction)*100,2)

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


model_names = []
exact_matches = []
f1_scores_mean = []
extended_matches = []
timing = []
best_epoch_number = []
best_batch_size = []
model_path_names = os.listdir("fine_tuning_results")
for model_path_name in model_path_names:
  model = pd.read_csv("/content/fine_tuning_results/" + model_path_name)
  model_name = model.iloc[-3]["predicted"]
  model_time = model.iloc[-3]["true"]
  model_epoch = model.iloc[-2]["predicted"]
  model_batch = model.iloc[-1]["predicted"]
  model = model.iloc[:-1]
  model_names.append(model_name)
  timing.append(model_time)
  best_epoch_number.append(model_epoch)
  best_batch_size.append(model_batch)
  model_predictions = model["predicted"].apply(lambda x:str(x))[:-2].tolist()
  model_actual_answers = model["true"].apply(lambda x:str(x))[:-2].tolist()
  extended_matches.append(extended_match(model_predictions,model_actual_answers))
  em_scores = [compute_exact(model_actual_answers[i],pred) for i,pred in enumerate(model_predictions)]
  f1_scores = [compute_f1(model_actual_answers[i],pred)*100 for i,pred in enumerate(model_predictions)]
  em_score = round(np.sum(em_scores)/len(em_scores)*100,2)
  f1_score = round(np.mean(f1_scores),2)
  exact_matches.append(em_score)
  f1_scores_mean.append(f1_score)

result_df = pd.DataFrame({"Pretrained Turkish BERT Models":model_names,"Exact Match":exact_matches,"F1-score":f1_scores_mean,"Extended Match":extended_matches,"Epoch":best_epoch_number,"Batch Size":best_batch_size,"Training Time":timing})
result_df.sort_values(by = "Exact Match",ascending = False,inplace = True)
result_df.to_csv("fine_tuned_scores.csv", index = False)
