# NLP Generative Pre-Training (GPT) Model
This repository includes the codes that I have modified for the GPT model, following the publication of [Open AI's GPT Model](https://openai.com/blog/better-language-models/). In particular, the changes include (i) the addition of a learnt positional embedding vector in each layer and (ii) the addition of a residual connection between the input embedding and the output embedding layer.

![GPT Model Architecture](GPT_network.png)

Fig. 1: GPT Model Architecture (obtained from [GPT paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf))

This repository includes a code to train the data on the [Movie Dialogue](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) dataset, where the preparation of the data follows this [script](https://github.com/suriyadeepan/datasets/blob/master/seq2seq/cornell_movie_corpus/scripts/prepare_data.py) closely. Instead of using a Sequence-to-Sequence model, the dialogue GPT model performs its inference by conditioning on the encoder inputs, followed by the `SOS` token to signal the beginning of the decoder output. For this model, the vocabulary is shared so the token embeddings are the same for both the encoder and decoder.

Before training the model, first process the data by running
```
python process_movie_dialogue.py
```
followed by
```
python movie_dialogue_tf_ver2_gpt.py
```
to train the GPT model. Run the script
```
python movie_dialogue_tf_ver2_gpt_test.py
```
to perform inference.

For the movie dialogue dataset, the training is done in the following manner - the encoder input is first inserted, followed by the `SOS token`, followed by the decoder output and finally ending with the `EOS token`. For example, if we have
```
Input Phrase:
how are you ?
Output Phrase:
SOS_token i ' m fine . EOS_token
```
as the encoder-decoder training pair, the GPT model will transform the encoder and decoder responses into
```
how are you ? SOS i ' m fine . EOS
```
and train the model on the concatenated response. Using the same example, inference is done by setting the seed as the encoder input followed by the `SOS_token`, or
```
how are you ? SOS
```
and for illustration purposes, if the GPT model's prediction of the entire sequence is
```
how are you ? SOS i am feeling fine . EOS
```
the decoder response will be taken as the GPT model's prediction following the `SOS_token`, which is `i am feeling fine . EOS`.

## Outputs
The GPT model on the movie dialogue dataset using a Nvidia Quadro P1000 4GB Graphics Card for 20000 iterations. Below are some sample outputs.
```
Input Phrase:
hello
Generated Response:
hi EOS PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD

Input Phrase:
how much does it cost
Generated Response:
thirty dollars EOS PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD

Input Phrase:
who is it
Generated Response:
it s me rock EOS PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD
```

### PyTorch Support
The relevant modules have been written using PyTorch as well. Run
```
python movie_dialogue_torch_gpt.py
```
to train the model using the PyTorch library. The data is pre-processed in the same manner as its Tensorflow counterpart.
