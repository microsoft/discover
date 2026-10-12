

import random
from random import shuffle

import argparse

from cfg_gen import *
from to_past import *

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", help="prefix of file to save the sequences to", type=str, default="base_sentences/sentences")
parser.add_argument("--cfg", help="file containing the CFG that defines the grammar to sample from", type=str, default="cfg.txt")
args = parser.parse_args()


cfg = fi_to_cfg(args.cfg)



n_train = 200000
n_valid = 5000
n_test = 5000

sentences = []
sentences_dict = {}

vocab = {}

while len(sentences) < n_train + n_valid + n_test:

    sentence = generate(cfg)
    words = sentence.lower().replace(".", "").replace(",", "").split()
    content_words = []
    for word in words:
        vocab[word] = 1
        if word not in ["the", "who", ".", "by", "near", "next", "to", "in", "front", "of", "behind", "across", "from"]:
            content_words.append(word)

    if len(list(set(content_words))) != len(content_words):
        # No repeats of content words
        continue


    if sentence not in sentences_dict:
        sentences_dict[sentence] = 1
        sentences.append([to_past(sentence), to_past(sentence), sentence])

        if len(sentences) % 10000 == 0:
            print(len(sentences))


# Check if vocab is all 1 token
model_name = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

for word in vocab:
    tokens = tokenizer([" " + word], return_tensors="pt")["input_ids"][0].tolist()
    if len(tokens) > 1:
        print(word)
        print(tokens)
        print(" ")


shuffle(sentences)

train_set = sentences[:n_train]
valid_set = sentences[n_train:n_train+n_valid]
test_set = sentences[n_train+n_valid:]

fo_train = open(args.prefix + ".train", "w")
fo_valid = open(args.prefix + ".valid", "w")
fo_test = open(args.prefix + ".test", "w")


fo_train.write("sentence\ttarget_seq\tanswer\n")
for line in train_set:
    fo_train.write("\t".join(line) + "\n")

fo_valid.write("sentence\ttarget_seq\tanswer\n")
for line in valid_set:
    fo_valid.write("\t".join(line) + "\n")

fo_test.write("sentence\ttarget_seq\tanswer\n")
for line in test_set:
    fo_test.write("\t".join(line) + "\n")









