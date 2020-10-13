import sys
import os
import re

all_txt_files = []
stopwords = set()
tokens = set()


def read_data():
    train_path = sys.argv[1]
    for root_path, dirs, files in os.walk(train_path):
        print("root_path:", root_path)
        if (re.search('positive', root_path.lower()) or re.search('negative', root_path.lower())) \
                and (re.search('truthful', root_path.lower()) or re.search('deceptive', root_path.lower())):
            for f in files:
                if f.endswith(".txt") and f != "README.txt":
                    all_txt_files.append(os.path.join(root_path, f))


def create_stopwords():
    f = open("stopwords.txt")
    for line in f:
        stopwords.add(line.strip())
    f.close()


def tokenization():
    for curr in all_txt_files:
        f = open(curr, "r")
        content = f.read()
        clean_content = re.sub(r'[^\w\s]', '', content).lower()
        word_arr = clean_content.split()
        for word in word_arr:
            if word not in stopwords:
                tokens.add(word)
        f.close()


def create_tf_matrix():
    positive = False
    truth = False

    positive_truth_docs = 0
    positive_deceptive_docs = 0
    negative_truth_docs = 0
    negative_deceptive_docs = 0

    positive_truth_wordcount = 0
    positive_deceptive_wordcount = 0
    negative_truth_wordcount = 0
    negative_deceptive_wordcount = 0

    filler = [1] * len(tokens)
    positive_truth_tf = dict(zip(tokens, filler))
    positive_deceptive_tf = dict(zip(tokens, filler))
    negative_truth_tf = dict(zip(tokens, filler))
    negative_deceptive_tf = dict(zip(tokens, filler))

    for doc_path in all_txt_files:
        if re.search('positive', doc_path.lower()):
            positive = True
            if re.search('truth', doc_path.lower()):
                truth = True
                positive_truth_docs += 1
                f = open(doc_path, "r")
                content = f.read()
                clean_content = re.sub(r'[^\w\s]', '', content).lower()
                word_arr = clean_content.split()
                for word in word_arr:
                    if word not in stopwords and word in positive_truth_tf:
                        positive_truth_wordcount += 1
                        positive_truth_tf[word] = positive_truth_tf.get(word) + 1
                f.close()
            elif re.search('deceptive', doc_path.lower()):
                truth = False
                positive_deceptive_docs += 1
                f = open(doc_path, "r")
                content = f.read()
                clean_content = re.sub(r'[^\w\s]', '', content).lower()
                word_arr = clean_content.split()
                for word in word_arr:
                    if word not in stopwords and word in positive_deceptive_tf:
                        positive_deceptive_wordcount += 1
                        positive_deceptive_tf[word] = positive_deceptive_tf.get(word) + 1
                f.close()
        elif re.search('negative', doc_path.lower()):
            positive = False
            if re.search('truth', doc_path.lower()):
                truth = True
                negative_truth_docs += 1
                f = open(doc_path, "r")
                content = f.read()
                clean_content = re.sub(r'[^\w\s]', '', content).lower()
                word_arr = clean_content.split()
                for word in word_arr:
                    if word not in stopwords and word in negative_truth_tf:
                        negative_truth_wordcount += 1
                        negative_truth_tf[word] = negative_truth_tf.get(word) + 1
                f.close()
            if re.search('deceptive', doc_path.lower()):
                truth = False
                negative_deceptive_docs += 1
                f = open(doc_path, "r")
                content = f.read()
                clean_content = re.sub(r'[^\w\s]', '', content).lower()
                word_arr = clean_content.split()
                for word in word_arr:
                    if word not in stopwords and word in negative_deceptive_tf:
                        negative_deceptive_wordcount += 1
                        negative_deceptive_tf[word] = negative_deceptive_tf.get(word) + 1
                f.close()

    positive_truth_wordcount += len(positive_truth_tf)
    positive_deceptive_wordcount += len(positive_deceptive_tf)
    negative_truth_wordcount += len(negative_truth_tf)
    negative_deceptive_wordcount += len(negative_deceptive_tf)

    output_f = open("nbmodel.txt", "w")

    output_f.write("positive_truth_docs:" + str(positive_truth_docs) + "\n")
    output_f.write("positive_deceptive_docs:" + str(positive_deceptive_docs) + "\n")
    output_f.write("negative_truth_docs:" + str(negative_truth_docs) + "\n")
    output_f.write("negative_deceptive_docs:" + str(negative_deceptive_docs) + "\n")

    output_f.write("positive_truth_wordcount:" + str(positive_truth_wordcount) + "\n")
    output_f.write("positive_deceptive_wordcount:" + str(positive_deceptive_wordcount) + "\n")
    output_f.write("negative_truth_wordcount:" + str(negative_truth_wordcount) + "\n")
    output_f.write("negative_deceptive_wordcount:" + str(negative_deceptive_wordcount) + "\n")

    output_f.write("POSITIVE_TRUTH\n")
    for key, value in positive_truth_tf.items():
        output_f.write(key + ":" + str(value) + "\n")

    output_f.write("POSITIVE_DECEPTIVE\n")
    for key, value in positive_deceptive_tf.items():
        output_f.write(key + ":" + str(value) + "\n")

    output_f.write("NEGATIVE_TRUTH\n")
    for key, value in negative_truth_tf.items():
        output_f.write(key + ":" + str(value) + "\n")

    output_f.write("NEGATIVE_DECEPTIVE\n")
    for key, value in negative_deceptive_tf.items():
        output_f.write(key + ":" + str(value) + "\n")


if __name__ == "__main__":
    read_data()
    create_stopwords()
    tokenization()
    create_tf_matrix()
