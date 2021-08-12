import time
import pandas as pd
import pickle as pkl
from collections import Counter
from nltk.tokenize import wordpunct_tokenize

# Define the function to clean the data. #
def clean_data(data_df, seq_len=9999):
    n_lines = len(data_df)
    
    # Process the data. #
    w_counter = Counter()
    data_list = []
    for n_line in range(n_lines):
        data_line = data_df.iloc[n_line]
        data_text = data_line["text"]
        data_label = data_line["label"]
        
        # Clean the data a little. #
        data_text = data_text.replace("\n", " ")
        
        # Tokenize the words. #
        tmp_tokens = [
            x for x in wordpunct_tokenize(
                data_text.lower()) if x != ""]
        
        w_counter.update(tmp_tokens)
        data_list.append((data_label, tmp_tokens))
        
        if (n_line+1) % 100000 == 0:
            percent_complete = round(n_line / n_lines * 100, 2)
            print(str(n_line), "rows", 
                  "(" + str(percent_complete) + "%) complete.")
    return data_list, w_counter

# Begin processing. #
start_time = time.time()
tmp_path = "C:/Users/admin/Desktop/Data/"
tmp_path += "Yelp/yelp_review_polarity_csv/"

print("Loading the data.")
train_df = pd.read_csv(tmp_path + "train.csv")
train_df.columns = ["label", "text"]
maximum_seq_len  = 100

# Process the train dataset. #
print("Cleaning the train data.")
train_data, w_counter = clean_data(
    train_df, seq_len=maximum_seq_len)

print("Data formatted. Total of", 
      str(len(train_data)), "training samples.")

# Filter noise. #
min_count  = 10
word_vocab = list(sorted([
    word for word, count in \
    w_counter.items() if count >= min_count]))
word_vocab = \
    ["CLS", "UNK", "PAD", "EOS", "TRUNC"] + word_vocab
idx_2_word = dict([(
    x, word_vocab[x]) for x in range(len(word_vocab))])
word_2_idx = dict([(
    word_vocab[x], x) for x in range(len(word_vocab))])

# Save the data. #
print("Saving the processed train data.")

tmp_pkl_file = tmp_path + "train_bert_data.pkl"
with open(tmp_pkl_file, "wb") as tmp_file_save:
    pkl.dump(train_data, tmp_file_save)
    pkl.dump(word_vocab, tmp_file_save)
    pkl.dump(idx_2_word, tmp_file_save)
    pkl.dump(word_2_idx, tmp_file_save)

elapsed_time = (time.time() - start_time) / 60.0
print("Elapsed time:", str(round(elapsed_time, 2)), "minutes.")

# For the test dataset. #
print("Processing Test Data.")

test_df = pd.read_csv(tmp_path + "test.csv")
test_df.columns = ["label", "text"]

# Process the test dataset. #
print("Cleaning the test data.")
test_data = clean_data(test_df)[0]

# Save the file. #
print("Saving the processed test data.")

tmp_pkl_file = tmp_path + "test_bert_data.pkl"
with open(tmp_pkl_file, "wb") as tmp_file_save:
    pkl.dump(test_data, tmp_file_save)
    pkl.dump(word_vocab, tmp_file_save)
    pkl.dump(idx_2_word, tmp_file_save)
    pkl.dump(word_2_idx, tmp_file_save)

print("Word vocabulary size:", str(len(word_vocab)))
print("Finish.")
