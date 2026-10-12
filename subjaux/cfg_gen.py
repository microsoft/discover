

import argparse
import random


def contains_nonterminal(seq, cfg):
    for elt in seq:
        if elt in cfg:
            return True

    return False


def generate(cfg, start="S"):

    seq = [start]


    while contains_nonterminal(seq, cfg):
        new_seq = []
        for index, elt in enumerate(seq):
            if elt in cfg:
                possible_rhs = cfg[elt]
                things = [x[0] for x in possible_rhs]
                things = [x[1] for x in possible_rhs]
                rhs = random.choices(population=[x[1] for x in possible_rhs], k=1, weights=[x[0] for x in possible_rhs])[0]
                new_seq = new_seq[:] + rhs
            else:
                new_seq.append(elt)

        seq = new_seq[:]

    sentence = " ".join(seq)
    sentence = sentence.replace(" ,", ",").replace(" .", ".")
    sentence = sentence[0].upper() + sentence[1:]

    return sentence



def fi_to_cfg(cfg_filename):
    fi = open(cfg_filename, "r")
    cfg = {}

    for line in fi:
        if line.strip().startswith("#"):
            continue
        elif len(line.strip()) == 0:
            continue

        parts = line.strip().split()
        weight = float(parts[0])
        lhs = parts[1]
        rhs = parts[2:]

        if lhs not in cfg:
            cfg[lhs] = []

        cfg[lhs].append([weight, rhs])

    fi.close()

    for lhs in cfg:
        total_weight = 0
        for weight, rhs in cfg[lhs]:
            total_weight += weight

        new_expansions = []
        for weight, rhs in cfg[lhs]:
            new_expansions.append([weight*1.0/total_weight, rhs])

        cfg[lhs] = new_expansions

    return cfg






