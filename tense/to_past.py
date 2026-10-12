

p2p = {}

p2p["recommend"] = "recommended"
p2p["visit"] = "visited"
p2p["help"] = "helped"
p2p["support"] = "supported"
p2p["surprise"] = "surprised"
p2p["hire"] = "hired"
p2p["avoid"] = "avoided"
p2p["advise"] = "advised"
p2p["stop"] = "stopped"
p2p["introduce"] = "introduced"
p2p["mention"] = "mentioned"
p2p["encourage"] = "encouraged"
p2p["thank"] = "thanked"
p2p["recognize"] = "recognized"
p2p["admire"] = "admired"

p2p["recommends"] = "recommended"
p2p["visits"] = "visited"
p2p["helps"] = "helped"
p2p["supports"] = "supported"
p2p["surprises"] = "surprised"
p2p["hires"] = "hired"
p2p["avoids"] = "avoided"
p2p["advises"] = "advised"
p2p["stops"] = "stopped"
p2p["introduces"] = "introduced"
p2p["mentions"] = "mentioned"
p2p["encourages"] = "encouraged"
p2p["thanks"] = "thanked"
p2p["recognizes"] = "recognized"
p2p["admires"] = "admired"


s2p = {}
s2p["recommends"] = "recommend"
s2p["visits"] = "visit"
s2p["helps"] = "help"
s2p["supports"] = "support"
s2p["surprises"] = "surprise"
s2p["hires"] = "hire"
s2p["avoids"] = "avoid"
s2p["advises"] = "advise"
s2p["stops"] = "stop"
s2p["introduces"] = "introduce"
s2p["mentions"] = "mention"
s2p["encourages"] = "encourage"
s2p["thanks"] = "thank"
s2p["recognizes"] = "recognize"
s2p["admires"] = "admire"

p2s = {}
for word in s2p:
    p2s[s2p[word]] = word

def toggle_number(verb):
    if verb in s2p:
        return s2p[verb]
    else:
        return p2s[verb]



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



def to_past(sentence):

    words = simple_tokenize(sentence)

    pos_tags = sent2pos(words)

    new_words = []
    for word in words:
        if word in p2p:
            new_words.append(p2p[word])
        else:
            new_words.append(word)
    new_sentence = " ".join(new_words).replace(" .", ".")
    new_sentence = new_sentence.capitalize()


    return new_sentence


#The intelligent manager who support the explorers stop the photographers .
def correct_number(words, pos_tags):

    corrected_words = []

    subj_orig_num = None
    recent_subj_num = None
    in_pp = False
    in_rc = False

    for index, (word, pos_tag) in enumerate(zip(words, pos_tags)):

        mismatch = False

        if pos_tag == "Nsg":
            if subj_orig_num is None:
                subj_orig_num = "sg"

            if not in_pp:
                recent_subj_num = "sg"

        if pos_tag == "Npl":
            if subj_orig_num is None:
                subj_orig_num = "pl"

            if not in_pp:
                recent_subj_num = "pl"




        if pos_tag == "P":
            in_pp = True

        if in_pp:
            if pos_tag in ["Nsg", "Npl"]:
                in_pp = False

        if pos_tag == "Comp":
            in_rc = True

        if pos_tag == "Vsg":
            if in_rc:
                if recent_subj_num == "pl":
                    mismatch = True
                in_rc = False

            else:
                if subj_orig_num == "pl":
                    mismatch = True


        if pos_tag == "Vpl":
            if in_rc:
                if recent_subj_num == "sg":
                    mismatch = True
                in_rc = False

            else:
                if subj_orig_num == "sg":
                    mismatch = True

        if mismatch:
            corrected_words.append(toggle_number(word))
        else:
            corrected_words.append(word)

    return corrected_words





if __name__ == "__main__":

    sentence = "The polite economist who supported the editor hired the judge who avoided the captain."
    print(sentence)
    print(passivize(sentence))

    print("")

    sentence = "The editor supported the captain."
    print(sentence)
    print(passivize(sentence))




