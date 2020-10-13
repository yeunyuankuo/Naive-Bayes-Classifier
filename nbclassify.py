import sys
import os
import re
import math
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

all_txt_files = []
predictions = []

positive_truth_tf = {}
positive_deceptive_tf = {}
negative_truth_tf = {}
negative_deceptive_tf = {}


def read_data():
    test_path = sys.argv[1]
    for root_path, dirs, files in os.walk(test_path):
        for f in files:
            if f.endswith(".txt") and f != "README.txt":
                all_txt_files.append(os.path.join(root_path, f))


def classify():
    positive_truth_docs = 0
    positive_deceptive_docs = 0
    negative_truth_docs = 0
    negative_deceptive_docs = 0

    positive_truth_wordcount = 0
    positive_deceptive_wordcount = 0
    negative_truth_wordcount = 0
    negative_deceptive_wordcount = 0

    f = open("nbmodel.txt", "r")
    positive_truth = False
    positive_deceptive = False
    negative_truth = False
    negative_deceptive = False
    for line in f:
        if line.strip() == "POSITIVE_TRUTH":
            positive_truth = True
            positive_deceptive = False
            negative_truth = False
            negative_deceptive = False
        elif line.strip() == "POSITIVE_DECEPTIVE":
            positive_truth = False
            positive_deceptive = True
            negative_truth = False
            negative_deceptive = False
        elif line.strip() == "NEGATIVE_TRUTH":
            positive_truth = False
            positive_deceptive = False
            negative_truth = True
            negative_deceptive = False
        elif line.strip() == "NEGATIVE_DECEPTIVE":
            positive_truth = False
            positive_deceptive = False
            negative_truth = False
            negative_deceptive = True
        else:
            detail = line.split(":")
            key = detail[0].strip()
            value = detail[1].strip()
            if key == "positive_truth_docs":
                positive_truth_docs = int(value)
            elif key == "positive_deceptive_docs":
                positive_deceptive_docs = int(value)
            elif key == "negative_truth_docs":
                negative_truth_docs = int(value)
            elif key == "negative_deceptive_docs":
                negative_deceptive_docs = int(value)
            elif key == "positive_truth_wordcount":
                positive_truth_wordcount = int(value)
            elif key == "positive_deceptive_wordcount":
                positive_deceptive_wordcount = int(value)
            elif key == "negative_truth_wordcount":
                negative_truth_wordcount = int(value)
            elif key == "negative_deceptive_wordcount":
                negative_deceptive_wordcount = int(value)
            if positive_truth:
                positive_truth_tf[key] = int(value)
            elif positive_deceptive:
                positive_deceptive_tf[key] = int(value)
            elif negative_truth:
                negative_truth_tf[key] = int(value)
            elif negative_deceptive:
                negative_deceptive_tf[key] = int(value)

    total_docs = positive_truth_docs + positive_deceptive_docs + negative_truth_docs + negative_deceptive_docs
    positive_truth_prior = math.log(positive_truth_docs / total_docs)
    positive_deceptive_prior = math.log(positive_deceptive_docs / total_docs)
    negative_truth_prior = math.log(negative_truth_docs / total_docs)
    negative_deceptive_prior = math.log(negative_deceptive_docs / total_docs)

    total_guess = 0
    correct_count = 0
    for curr in all_txt_files:
        total_guess += 1
        f = open(curr, "r")
        content = f.read()
        clean_content = re.sub(r'[^\w\s]', '', content).lower()
        word_arr = clean_content.split()

        '''1. positive_truth'''
        positive_truth_probability = positive_truth_prior
        for word in word_arr:
            if word in positive_truth_tf:
                positive_truth_probability += math.log(positive_truth_tf[word] / positive_truth_wordcount)

        '''2. positive_deceptive'''
        positive_deceptive_probability = positive_deceptive_prior
        for word in word_arr:
            if word in positive_deceptive_tf:
                positive_deceptive_probability += math.log(positive_deceptive_tf[word] / positive_deceptive_wordcount)

        '''3. negative_truth'''
        negative_truth_probability = negative_truth_prior
        for word in word_arr:
            if word in negative_truth_tf:
                negative_truth_probability += math.log(negative_truth_tf[word] / negative_truth_wordcount)

        '''4. negative_deceptive'''
        negative_deceptive_probability = negative_deceptive_prior
        for word in word_arr:
            if word in negative_deceptive_tf:
                negative_deceptive_probability += math.log(negative_deceptive_tf[word] / negative_deceptive_wordcount)

        if (positive_truth_probability >= positive_deceptive_probability
                and positive_truth_probability >= negative_truth_probability
                and positive_truth_probability >= negative_deceptive_probability):
            # print("PATH: ", curr)
            predictions.append("truthful positive " + curr)
            if re.search("positive", curr) and re.search("truth", curr):
                correct_count += 1
                # print('O - positive_truth')
            # else:
                # print('X - positive_truth')
        elif (positive_deceptive_probability >= positive_truth_probability
                and positive_deceptive_probability >= negative_truth_probability
                and positive_deceptive_probability >= negative_deceptive_probability):
            # print("PATH: ", curr)
            predictions.append("deceptive positive " + curr)
            if re.search("positive", curr) and re.search("deceptive", curr):
                correct_count += 1
                # print("O - positive_deceptive")
            # else:
                # print("X - positive_deceptive")
        elif (negative_truth_probability >= positive_truth_probability
                and negative_truth_probability >= positive_deceptive_probability
                and negative_truth_probability >= negative_deceptive_probability):
            # print("PATH: ", curr)
            predictions.append("truthful negative " + curr)
            if re.search("negative", curr) and re.search("truth", curr):
                correct_count += 1
                # print("O - negative_truth")
            # else:
                # print("X - negative_truth")
        elif (negative_deceptive_probability >= positive_truth_probability
                and negative_deceptive_probability >= positive_deceptive_probability
                and negative_deceptive_probability >= negative_truth_probability):
            # print("PATH: ", curr)
            predictions.append("deceptive negative " + curr)
            if re.search("negative", curr) and re.search("deceptive", curr):
                correct_count += 1
                # print("O - negative_deceptive")
            # else:
                # print("X - negative_deceptive")
        else:
            predictions.append("truthful negative " + curr)

    print("correct: ", correct_count, "/", total_guess)


def calculate_f1():
    true = []
    pred = []
    for doc in all_txt_files:
        if re.search("positive", doc) and re.search("truth", doc):
            true.append(0)
        elif re.search("positive", doc) and re.search("deceptive", doc):
            true.append(1)
        elif re.search("negative", doc) and re.search("truth", doc):
            true.append(2)
        elif re.search("negative", doc) and re.search("deceptive", doc):
            true.append(3)

    for curr in predictions:
        detail = curr.split()
        if detail[0] == "truthful" and detail[1] == "positive":
            pred.append(0)
        elif detail[0] == "deceptive" and detail[1] == "positive":
            pred.append(1)
        elif detail[0] == "truthful" and detail[1] == "negative":
            pred.append(2)
        elif detail[0] == "deceptive" and detail[1] == "negative":
            pred.append(3)

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(true, pred)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(true, pred, average="macro")
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(true, pred, average="macro")
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    print("f-1 score: ", f1_score(true, pred, average="macro"))


def write_output():
    output_f = open("nboutput.txt", "w")
    for curr in predictions:
        output_f.write(curr)
        output_f.write("\n")
    output_f.close()


if __name__ == "__main__":
    read_data()
    classify()
    calculate_f1()
    write_output()
