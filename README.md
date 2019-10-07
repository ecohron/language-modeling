# Language Modeling
Implements and trains a linearly interpolated language model.

## problem-writeup.pdf
Describes the original assignment goals and parameters.

## model.py
Contains a LanguageModel class which builds a language model from a set of training data. Also contains the calculate_perplexity method which calculates the perplexity of a given model.

## utils.py
Defines the model utils including read_file, load_dataset, and preprocess which preprocesses the corpus file from a string to a list of strings in which each index represents a different privacy policy.

## data
Contains five files each including 50 line-separated privacy policies of websites and companies. Each file contains the policies of companies of a different industry.
