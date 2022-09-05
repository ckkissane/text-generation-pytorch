PyTorch implementations of methods for text generation / sampling from GPT-2.
Includes methods like beam search, top k sampling, and nucleus sampling.

The implementations are meant to be simple and educational. 
They are written for HuggingFace's [GPT2LMHeadModel](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel), 
but can be tweaked to work with other models.

Summary of the different methods in this repo:

* Greedy search: Always choose the most likely next token.
This is simple and efficient, but empirically seems to cause repetitive and boring text generation.

* Beam search: Instead of only considering the next token, why not consider the conditional probability
of the entire generated sequence? It would be computationally intractable to try every possible 
sequence you can generate, but you can prune the search based on a beam width. 
This empirically leads to better results than greedy search, but is more computationally expensive for larger
beam widths.

* Sample: Sometimes we don't really care about high probability sequences, rather we just want to read interesting text. 
One way to add spontaneity is by sampling from the distribution. 
However, too much randomness can cause the model to sound incoherent.

* Temperature sampling: One trick to "sharpen" the probability distribution is to scale the logits
by a temperature value between 0 and 1. A well tuned temperature will cause the model to have
more conviction in its most likely options.

* Top k sampling: Even with temperature, we might eventually pick some very unlikely tokens that
derail our model.
Top k sampling filters out all but the top k most likely options, and then samples from the remaining. 

* Top p (nucleus) sampling: One issue with top k sampling is that "k" is fixed. 
Sometimes there might be a few clear options where we want small k. 
Other situations will have lots of viable options, where we prefer big k. 
Top p sampling solves this by computing the cumulative distribution and cutting it off
as the CDF exceeds p. This effectively auto tunes k based on the shape of the distribution.

Resources:
* [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)
* [How to sample from language models](https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277)
