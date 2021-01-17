# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:44:37 2020

@author: admin
"""

import json
import pandas as pd
import pickle as pkl
from collections import Counter
from nltk.tokenize import wordpunct_tokenize as word_tokenizer

# Open the json file. #
tmp_file = "C:/Users/admin/Desktop/Codes/reddit_jokes.json"
tmp_data = json.loads(open(tmp_file).read())

min_count  = 5
seq_length = 51

# Extract the data. #
tmp_list  = []
tmp_jokes = []
for tmp_row in tmp_data:
    if tmp_row["body"].find(tmp_row["title"]) != -1:
        tmp_joke = tmp_row["body"]
    elif tmp_row["title"].find(tmp_row["body"]) != -1:
        tmp_joke = tmp_row["title"]
    else:
        tmp_joke = tmp_row["title"] + " " + tmp_row["body"]
    tmp_list.append(( tmp_row["id"], tmp_row["score"], tmp_joke))
    
    if tmp_row["score"] < 12:
        tmp_class = "bad_joke"
    elif tmp_row["score"] < 50:
        tmp_class = "ok_joke"
    else:
        tmp_class = "good_joke"
    tmp_jokes.append(tmp_class + " " + tmp_joke)
del tmp_joke

# Convert into a DataFrame. #
tmp_df = pd.DataFrame(tmp_list, columns=["id", "scores", "joke"])
tmp_df.to_csv(
    "C:/Users/admin/Desktop/Codes/reddit_jokes.csv", index=False)
del tmp_list

# Process the data. #
tmp_list  = []
w_counter = Counter()
for tmp_joke in tmp_jokes:
    tmp_tokens = [
        x for x in word_tokenizer(tmp_joke.lower()) if x != ""]
    if len(tmp_tokens) <= seq_length:
        tmp_list.append(" ".join(tmp_tokens))
        w_counter.update(tmp_tokens)

vocab_size = 12500
full_vocab = sorted([
    x for x, y in w_counter.items() if y >= min_count])
vocab_list = sorted([
    x for x, y in w_counter.most_common(vocab_size) if y >= min_count])
vocab_list = ["SOS", "EOS", "PAD", "UNK"] + vocab_list

idx2word = dict([
    (x, vocab_list[x]) for x in range(len(vocab_list))])
word2idx = dict([
    (vocab_list[x], x) for x in range(len(vocab_list))])

print("Full Vocabulary Size:", str(len(w_counter)))
print("Thresh Vocabulary Size:", str(len(full_vocab)))
print("Trimmed Vocabulary size:", str(len(vocab_list)))
print("Total number of jokes:", str(len(tmp_list)))

# Save the data. #
tmp_pkl_file = \
    "C:/Users/admin/Desktop/Codes/reddit_jokes.pkl"
with open(tmp_pkl_file, "wb") as tmp_file_save:
    pkl.dump(tmp_list, tmp_file_save)
    pkl.dump(idx2word, tmp_file_save)
    pkl.dump(word2idx, tmp_file_save)

