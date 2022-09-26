# Sequence-to-Sequence Transformer Network

This repository contains an implementation of the [Transformer Network](https://arxiv.org/abs/1706.03762) with some minor changes - (i) positional embedding at each layer of the encoder and decoder is learnt, and (ii) the final output has a residual connection with its input embeddings. To account for limited compute resources, the training step breaks down the training batch into sub-batches and accumulates the gradient before averaging and updating the weights via the `sub_batch_train_step` function. Generally, it is observed that the Transformer Network's training is more stable with a batch size between 128 and 512. Sub-word tokenization is supported via `byte_pair_encoding.py` script. 

This model is trained on a few data sets, including:
* [Cornell Movie Dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
* [Tweet Dataset](https://github.com/Marsan-Ma-zz/chat_corpus)

For the Cornell Movie Dataset, the processing code follows this [script](https://github.com/suriyadeepan/datasets/blob/master/seq2seq/cornell_movie_corpus/scripts/prepare_data.py) closely.

To run the chatbot, set the paths and parameters in the various scripts accordingly before running it. For the movie dialogue dataset, run
```
python process_movie_dialogue.py
```
if word tokens are used, or
```
python process_movie_dialogue_subword.py
```
if sub-word tokens are used. After which, run
```
python movie_dialogue_transformer_tf_ver2.py
```
to train the chatbot on the movie dialogue dataset. For the Twitter dataset, the commands to run are
```
python process_twitter_corpus.py
```
and
```
python twitter_transformer_tf_ver2.py
```
respectively. An limit of 10 words/subwords at both the encoder and decoder is set but this can be modified. 

A large portion of this write-up was taken from my [Red Dragon NLP assignment](https://github.com/WD-Leong/Red-Dragon-AI-Course-Advanced-NLP-Assignment-2), and the slightly modified Transformer architecture is presented in this write-up. The hardware used is a Nvidia Quadro P1000 4GB Graphics card.

## Data Pre-processing
A simple pre-processing for the data was applied, including lower-casing, removing punctuation and applying word tokenisation to the text thereafter. Some examples of the processed data is shown below.
| Movie Dialogue Input | Movie Dialogue Output |
| -------------------- | --------------------- |
| gosh if only we could find kat a boyfriend | let me see what i can do |
| take it or leave it this isn t a negotiation | fifty and you ve got your man |
| are you ready to put him in | not yet |

| Twitter Dialogue Input | Twitter Dialogue Output |
| ---------------------- | ----------------------- |
| you want to turn twitter followers into blog readers | how do you do this |
| fooled them by over dubbing him with racist speech | what speech of his was racist |
| today already fuckin sucks | what s wrong |

Only single turn conversation is considered in the pre-processing step. Hence, a multi-turn conversation which involves a few rounds of dialogue between the parties is not considered.

## Generating the Vocabulary
To generate the vocabulary, certain symbols like `\\u`, `\\i`, as well as newlines `\n` and tabs `\t` are replaced in the movie dialogue dataset. The regular expression `re.sub(r"[^\w\s]", " ", tmp_qns)` was used to remove punctuations from the text for all datasets. It is possible to use `wordpunct_tokenize` from the `nltk` package to process the data as well.

For the movie dialogue dataset, the processing steps can be found in the snippet:
```
import re

tmp_qns = id2line[conv[i]].lower().replace(
    "\\u", " ").replace("\\i", " ").replace("\n", " ").replace("\t", " ")
tmp_qns = re.sub(r"[^\w\s]", " ", tmp_qns)
tmp_qns = [x for x in tmp_qns.split(" ") if x != ""]
```
For the Twitter dataset, the slightly different processing steps can be found in the snippet:
```
tmp_q = re.sub(r"[^\w\s]", " ", tmp_q)
tmp_q = [x for x in tmp_q.split(" ") if x != ""]
```
This is followed by updating the word token counter via
```
w_counter.update(tmp_qns)
w_counter.update(tmp_ans)
```
for the movie dialogue dataset where a joint vocabulary is used, or via
```
q_counter.update(tmp_q)
a_counter.update(tmp_a)
```
for the Twitter dataset where separate vocabularies for the encoder and decoder sections is used. The vocabulary is then updated by taking the most common `vocab_size` tokens via
```
vocab_size = 8000
vocab_list = sorted([x for x, y in w_counter.most_common(vocab_size)])
vocab_list = ["SOS", "EOS", "PAD", "UNK"] + vocab_list
```
where the special tokens `SOS`, `EOS`, `PAD` and `UNK` are added to the vocabulary. The joint vocabulary is given by
```
idx2word = dict([
    (x, vocab_list[x]) for x in range(len(vocab_list))])
word2idx = dict([
    (vocab_list[x], x) for x in range(len(vocab_list))])
```
or the separate vocabularies are given by 
```
q_vocab_list = sorted([
    x for x, y in q_counter.most_common(vocab_size)])
q_vocab_list = ["SOS", "PAD", "UNK"] + q_vocab_list
a_vocab_list = sorted([
    x for x, y in a_counter.most_common(vocab_size)])
a_vocab_list = ["SOS", "EOS", "PAD", "UNK"] + a_vocab_list

q_idx2word = dict([
    (x, q_vocab_list[x]) for x in range(len(q_vocab_list))])
q_word2idx = dict([
    (q_vocab_list[x], x) for x in range(len(q_vocab_list))])
a_idx2word = dict([
    (x, a_vocab_list[x]) for x in range(len(a_vocab_list))])
a_word2idx = dict([
    (a_vocab_list[x], x) for x in range(len(a_vocab_list))])
```
depending on the configuration that is desired.

## Our Transformer Model
This Transformer model has some slight deviations to the original model in that (i) it learns a positional embedding layer at each layer of the encoder and decoder, (ii) it adds a residual connection between the input embeddings at both the encoder and decoder, and (iii) applies layer normalisation before the residual connection.

### Transformer Parameters
The model uses 3 layers for both the encoder and decoder, an embedding dimension of 256, a hidden size of 256 and a feed-forward size of 1024. The sequence length at both the encoder and decoder was set to 10, which led to a total of approximately 94000 movie dialogue sequences, and a vocabulary of the most common 8000 words was used. A dropout probability of 0.1 was set to allow the model to generalise and the gradient was clipped at 1.0 during training. The loss function is the cross-entropy loss function and the Adam optimizer was used to perform the weight updates.

Before sending the data into the Transformer model, the dialogue sequences need to be converted into their corresponding integer labels. This is done via
```
tmp_i_tok = data_tuple[tmp_index][0].split(" ")
tmp_o_tok = data_tuple[tmp_index][1].split(" ")

tmp_i_idx = [word2idx.get(x, UNK_token) for x in tmp_i_tok]
tmp_o_idx = [word2idx.get(x, UNK_token) for x in tmp_o_tok]
```
where `word2idx` is a dictionary mapping the word tokens into their corresponding integer labels. If the sequence is shorter than the encoder length, the remainder of the sequence is padded with the `PAD` token.

The model has approximately 57 million parameters as returned by `seq2seq_model.summary()` function. As our model is encapsulated in a custom class and applied without using the standard `tf.keras` functionalities, the summary of the model is unable to breakdown the number of parameters at each step.
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
Total params: 21,095,168
Trainable params: 21,095,168
Non-trainable params: 0
_________________________________________________________________
```
Due to limitations on the GPU card, we accumulate the gradients manually across sub-batches of 64, then average it to apply the overall weight update across a larger batch of 256 examples. 
```
with tf.GradientTape() as grad_tape:
    output_logits = model(tmp_encode, tmp_decode)
    
    tmp_losses = tf.reduce_sum(tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tmp_output, logits=output_logits), axis=1))
    
    # Accumulate the gradients. #
    tot_losses += tmp_losses
    tmp_gradients = \
        grad_tape.gradient(tmp_losses, model_params)
    acc_gradients = [
        (acc_grad+grad) for \
        acc_grad, grad in zip(acc_gradients, tmp_gradients)]
```
Following the [T5 paper](https://arxiv.org/abs/1910.10683), 2000 warmup steps with a constant learning rate was applied `step_val = float(max(n_iter+1, warmup_steps))**(-0.5)`.

## Inference with the Transformer
Greedy decoding is implemented in the `infer` function within the `TransformerNetwork` class.
```
def infer(self, x_encode, x_decode):
    # Encoder. #
    eps = 1.0e-6
    x_enc_token = tf.nn.embedding_lookup(self.W_enc_emb, x_encode)
    x_dec_token = tf.nn.embedding_lookup(self.W_dec_emb, x_decode)
    
    ...
    
    enc_outputs = transformer_encode(...)
    
    # Inference. #
    infer_embed = [tf.expand_dims(x_dec_token[:, 0, :], axis=1)]
    infer_index = []
    for step in range(self.seq_decode):
        ...
        
        tmp_outputs = transformer_decode(...)
        
        tmp_logit  = tf.matmul(
            tmp_outputs[:, -1, :], self.p_out_decode)
        tmp_argmax = tf.argmax(
            tmp_logit, axis=-1, output_type=tf.int32)
        next_embed = tf.matmul(
            tf.nn.softmax(tmp_logit), self.W_dec_emb)
        
        infer_index.append(tf.expand_dims(tmp_argmax, axis=1))
        infer_embed.append(tf.expand_dims(next_embed, axis=1))
        
    # Concatenate the list into a tensor. #
    infer_ids = tf.concat(infer_index, axis=1)
    return infer_ids
```
Nonetheless, the inference uses a softmax to obtain a weighted embedding representation of the next input via
```
next_embed = tf.matmul(tf.nn.softmax(tmp_logit), self.W_dec_emb)
```
to reduce the extent of error propagation caused by incorrect prediction of any label during the inference process.

## Training Progress of the Transformer Network
As the training progressed, the Transformer network's response was logged and presented in this section.
```
--------------------------------------------------
Iteration 5000:
Average Loss: 19.312504348754878

Input Phrase:
what is he doing there
Generated Phrase:
he didn EOS PAD PAD PAD PAD PAD PAD PAD PAD
Actual Response:
he appears to be standing by his horse
--------------------------------------------------
Iteration 10000:
Average Loss: 16.14546756744385

Input Phrase:
how do you know
Generated Phrase:
i don know EOS PAD PAD PAD PAD PAD PAD PAD
Actual Response:
i don t know i m just guessing
--------------------------------------------------
Iteration 20000:
Average Loss: 11.604991626739501

Input Phrase:
oh can i help you
Generated Phrase:
i don mr you my woody and yet it EOS PAD
Actual Response:
well that depends do you have a bathroom
--------------------------------------------------
Iteration 25000:
Average Loss: 10.241722373962402

Input Phrase:
yes it reminds me of home
Generated Phrase:
is EOS PAD PAD PAD PAD PAD PAD PAD PAD PAD
Actual Response:
where superman s from krypton
--------------------------------------------------
Iteration 30000:
Average Loss: 9.130070762634277

Input Phrase:
i m fine still working with my father
Generated Phrase:
you don hold right tonight EOS tom you father EOS PAD
Actual Response:
and what does he do again
--------------------------------------------------
```

![tranformer_learning](train_progress_transformer_dialogue.png)

Fig. 1: Training Progress of the Chatbot

## Testing the Dialogue Transformer Network Chatbot
Having trained the chatbot, we can now try out some sample responses using the `movie_dialogue_test.py` script. While in an actual scenario, the replies after the `EOS` token should be disregarded, we display the entire response of the trained model.
```
--------------------------------------------------
Enter your phrase: hello

Input Phrase:
hello
Generated Phrase:
hi EOS 
--------------------------------------------------
Enter your phrase: how are you

Input Phrase:
how are you
Generated Phrase:
fine thank EOS
--------------------------------------------------
Enter your phrase: what time is it

Input Phrase:
what time is it
Generated Phrase:
twelve o clock EOS
--------------------------------------------------
Enter your phrase: how much does it cost

Input Phrase:
how much does it cost
Generated Phrase:
five dollars EOS
--------------------------------------------------
Enter your phrase: what did he tell you

Input Phrase:
what did he tell you
Generated Phrase:
he didn t hear it nothin EOS
--------------------------------------------------
Enter your phrase: goodbye

Input Phrase:
goodbye
Generated Phrase:
goodbye EOS PAD PAD PAD PAD PAD PAD PAD PAD PAD
--------------------------------------------------
```
As we can observe, the chatbot appears to generate relatively proper responses. It is also able to understand and reply when asked questions relating to time or place, as seen from its responses to `what time is it` and `how much does it cost`. The slang can also be observed to be learnt, as seen from the reply `he didn t hear it nothin`.

## PyTorch Support
The relevant modules have been written using PyTorch as well. Run
```
python movie_dialogue_torch_transformer.py
```
to train the model using the PyTorch library, and
```
python movie_dialogue_torch_transformer_test.py
```
to do inference. The data is pre-processed in the same manner as its Tensorflow counterpart.

The original model in `torch_transformer_network.py` has been refactored to use `torch.nn.Module` in `torch_transformer.py`. To train the model, run
```
python train_movie_dialog_sw_torch_transformer.py
```
and run the command
```
python infer_movie_dialog_sw_torch_transformer
```
to perform inference after the model is trained.

## Trained Models
Some of the trained models can be downloaded via:

| Model Name | Link |
| ---------- | ---- |
| PyTorch Dialogue Word Seq2Seq Transformer | https://github.com/WD-Leong/NLP-Transformer/releases/download/v0.9/transformer_seq2seq |
| TensorFlow v2 Dialogue Seq2Seq Transformer | https://github.com/WD-Leong/NLP-Transformer/releases/download/v0.9/transformer_seq2seq.zip |

The data can be downloaded via:

| Data Type | Link |
| --------- | ---- |
| Movie Dialogue Word Data | https://github.com/WD-Leong/NLP-Transformer/releases/download/v0.9/movie_dialogues.pkl |
| Movie Dialogue Sub-word Data | https://github.com/WD-Leong/NLP-Transformer/releases/download/v0.9/movie_dialogues_subword.pkl |

## Conclusion
A dialogue chatbot using both the movie dialogue and Twitter dataset. Due to computation constraints, the encoder and decoder length were both set to 10, and the standard base Transformer network was used with some minor modifications. Depending on the scenario, it supports both byte-pair encoding as well as joint vocabulary. The modified Transformer model can be observed to provide proper replies in general. 
