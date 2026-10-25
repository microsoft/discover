
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


import random
from random import shuffle

import argparse

from cfg_gen import *
from passivize import *

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", help="prefix of file to save the sequences to", type=str, default="base_sentences/sentences")
parser.add_argument("--cfg", help="file containing the CFG that defines the grammar to sample from", type=str, default="cfg.txt")
args = parser.parse_args()


cfg = fi_to_cfg(args.cfg)


n_interventions = 100

sentences = []
sentences_dict = {}

vocab = {}

while len(sentences) < n_interventions:

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

    word_to_replace = random.choice(content_words)
    pos = sent2pos([word_to_replace])[0]
    new_word = generate(cfg, start=pos).lower()
    if new_word == word_to_replace:
        continue


    if sentence not in sentences_dict:
        sentences_dict[sentence] = 1
        new_sentence = sentence.replace(word_to_replace, new_word)
        sentences.append([sentence, sentence, passivize(sentence), new_sentence, passivize(new_sentence), "(DUMMY," + word_to_replace + ")", "(DUMMY," + new_word + ")"])

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

fo = open("base_sentences/interventions.content", "w")
fo_eval = open("base_sentences/interventions_eval.content", "w")

fo.write("sentence\ttarget_seq\tanswer\tnew_sentence\tnew_answer\tfr_pairs_old\tfr_pairs_new\n")
fo_eval.write("sentence\ttarget_seq\tanswer\n")
for line in sentences:
    fo.write("\t".join(line) + "\n")
    fo_eval.write(line[3] + "\t" + line[3] + "\t" + line[4] + "\n")
fo.close()
fo_eval.close()





