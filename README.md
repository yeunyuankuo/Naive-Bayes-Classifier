# Naive Bayes Classifier
This is a Naive Bayes Classifier hand written from scratch using only Python's built in library (no pre-written third party packages or libraries are used).

## Overview
This Naive Bayes Classifier will be used to identify hotel reviews (in english texts) as either truthful(actual reviews from web) or deceptive(made-up fake reviews), and either positive or negative. I will be using work tokens as features for classification.

## Data
- A top-level directory with two sub-directories, one for positive reviews and another for negative reviews.
- Each of the subdirectories contains two sub-directories, one with truthful reviews and one with deceptive reviews.
- Each of these subdirectories contains four subdirectories, called “folds”.
- Each of the folds contains 80 text files with English text (one review per file).

Data in the folder named "train" will be used to train the naive bayes model and the data in the folder named "test"" will be used to evaluate the model.

## How to run the program
### Training The Model:
The learning or training part of the program will be invoked in the following way:
> python3 nblearn.py /path/to/input

The argument is the directory of the training data; the program will learn a Naive Bayes model, and write the model parameters to a file called nbmodel.txt

1. The model file should contain sufficient information for nbclassify.py to successfully label new data.
2. The model file should be human-readable, so that model parameters can be easily understood by visual inspection of the file.

### Classification Part:
The classification or prediction part will be invoked in the following way:
> python3 nbclassify.py /path/to/input

The argument is the directory of the test data; the program will read the parameters of a naive Bayes model from the file nbmodel.txt, classify each file in the test data, and write the results to a text file called nboutput.txt in the following format:

label_a label_b path1

label_a label_b path2

...


In the above format, label_a is either “truthful” or “deceptive”, label_b is either “positive” or “negative”, and pathn is the path of the text file being classified.
