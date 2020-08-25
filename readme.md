# Sequence-to-Sequence Transformer Network

This repository contains my implementation of the [Transformer Network](https://arxiv.org/abs/1706.03762) with some minor changes - (i) positional embedding at each layer of the encoder and decoder is learnt, and (ii) the final output has a residual connection with its input embeddings. To account for limited compute resources, the training step breaks down the training batch into sub-batches and accumulates the gradient before averaging and updating the weights via the `sub_batch_train_step` function. Generally, I observed that the Transformer Network's training is more stable with a batch size between 128 and 512. Sub-word tokenization is supported via `byte_pair_encoding.py` script. 

This model is trained on a few data sets, including:
* [Cornell Movie Dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
* [Tweet Dataset](https://github.com/Marsan-Ma-zz/chat_corpus)

For the Cornell Movie Dataset, the processing code follows this [script](https://github.com/suriyadeepan/datasets/blob/master/seq2seq/cornell_movie_corpus/scripts/prepare_data.py) closely.

To run the chatbot, set the paths and parameters in the various scripts accordingly before running it. For the movie dialogue dataset, run
```
python process_movie_dialogue.py
```
followed by
```
python movie_dialogue_transformer_tf_ver2.py
```
to train the chatbot on the movie dialogue dataset. An input limit of 10 words/subwords at both the encoder and decoder is set but this can be modified. 
