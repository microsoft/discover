

def remove_excess_space(string):
    if "  " in string:
        return remove_excess_space(string.replace("  ", " "))
    else:
        return string

def simple_tokenize(sent):
    sent = remove_excess_space(sent.replace(".", " . ").replace(",", " , ").strip()).lower()

    return sent.split()


word2pos = {}
fi = open("cfg.txt", "r")
for line in fi:
    parts = line.strip().split()
    if len(parts) >= 3:
        words = parts[2:]
        for word in words:
            word2pos[word] = parts[1]
fi.close()


def sent2pos(words):
    pos_tags = []
    for word in words:
        pos_tags.append(word2pos[word])

    return pos_tags



def invert(sentence):

    words = simple_tokenize(sentence)

    pos_tags = sent2pos(words)

    ind_main_verb = -1

    seen_wh = False
    main_verb_index = None
    for index, (word, pos_tag) in enumerate(zip(words, pos_tags)):
        if pos_tag == "Comp":
            seen_wh = True

        if pos_tag == "V":
            if seen_wh:
                seen_wh = False
            elif main_verb_index is None:
                main_verb_index = index

    main_aux = words[main_verb_index-1]
    pre = words[:main_verb_index-1]
    post = words[main_verb_index:-1]

    new_sentence = [main_aux] + pre + post
    new_sentence = " ".join(new_sentence) + "?"
    new_sentence = new_sentence.capitalize()


    return new_sentence



if __name__ == "__main__":

    sentence = "The polite economist who supported the editor hired the judge who avoided the captain."
    print(sentence)
    print(passivize(sentence))

    print("")

    sentence = "The editor supported the captain."
    print(sentence)
    print(passivize(sentence))




