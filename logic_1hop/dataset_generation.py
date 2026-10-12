

import random

from transformers import AutoTokenizer


# a: all A are B
# e: no A are B
# i: some A are B
# o: some A are not B
#
# Figure 1: B C, A B -> A C. 
# Figure 2: C B, A B -> A C
# Figure 3: B C, B A -> A C
# Figure 4: C B, B A -> A C

possible_syllogisms = [["All Y are Z.\\nAll X are Y.", "All X are Z."],
    ["No Y are Z.\\nAll X are Y.", "No X are Z."],
    ["All Y are Z.\\nSome X are Y.", "Some X are Z."],
    ["No Y are Z.\\nSome X are Y.", "Some X are not Z."],
    ["All Y are Z.\\nAll X are Y.", "Some X are Z."],
    ["No Y are Z.\\nAll X are Y.", "Some X are not Z."],

    ["All Z are Y.\\nNo X are Y.", "No X are Z."],
    ["No Z are Y.\\nAll X are Y.", "No X are Z."],
    ["No Z are Y.\\nSome X are Y.", "Some X are not Z."],
    ["All Z are Y.\\nSome X are not Y.", "Some X are not Z."],
    ["All Z are Y.\\nNo X are Y.", "Some X are not Z."],
    ["No Z are Y.\\nAll X are Y.", "Some X are not Z."],

    ["All Y are Z.\\nSome Y are X.", "Some X are Z."],
    ["Some Y are Z.\\nAll Y are X.", "Some X are Z."],
    ["Some Y are not Z.\\nAll Y are X.", "Some X are not Z."],
    ["No Y are Z.\\nSome Y are X.", "Some X are not Z."],
    ["All Y are Z.\\nAll Y are X.", "Some X are Z."],
    ["No Y are Z.\\nAll Y are X.", "Some X are not Z."],

    ["All Z are Y.\\nNo Y are X.", "No X are Z."],
    ["Some Z are Y.\\nAll Y are X.", "Some X are Z."],
    ["No Z are Y.\\nSome Y are X.", "Some X are not Z."],
    ["All Z are Y.\\nNo Y are X.", "Some X are not Z."],
    ["No Z are Y.\\nAll Y are X.", "Some X are not Z."],
    ["All Z are Y.\\nAll Y are X.", "Some X are Z."]
]



syllogism_dict = {}
for lhs, rhs in possible_syllogisms:
    if lhs not in syllogism_dict:
        syllogism_dict[lhs] = []
    syllogism_dict[lhs].append(rhs)

strongest_syllogism_dict = {}
for lhs in syllogism_dict:
    if len(syllogism_dict[lhs]) == 1:
        strongest_syllogism_dict[lhs] = syllogism_dict[lhs][0]
    elif len(syllogism_dict[lhs]) == 2:
        rhses = syllogism_dict[lhs]
        if rhses == ["All X are Z.", "Some X are Z."]:
            strongest_syllogism_dict[lhs] = "All X are Z."
        elif rhses == ["Some X are Z.", "All X are Z."]:
            strongest_syllogism_dict[lhs] = "All X are Z."
        elif rhses == ["No X are Z.", "Some X are not Z."]:
            strongest_syllogism_dict[lhs] = "No X are Z."
        elif rhses == ["Some X are not Z.", "No X are Z."]:
            strongest_syllogism_dict[lhs] = "No X are Z."
        else:
            print(lhs, syllogism_dict[lhs])
    else:
        print(lhs, syllogism_dict[lhs])


# opticians -> dentists
# cashiers -> accountants
# clerks -> administrators
# tellers -> teachers
# linguists -> professors
# chemists -> lawyers
# porters -> senators
# butchers -> musicians
# barbers -> presidents
# novelists -> philosophers
# florists -> artists
# skaters -> divers
# joggers -> guards
# carpenters -> engineers
# waiters -> servers
# tailors -> directors
# sculptors -> tutors
# actuaries -> psychologists
word_triples = [["psychologists", "tutors", "writers"],
        ["lawyers", "drivers", "dancers"],
        ["dancers", "bankers", "riders"],
        ["farmers", "surfers", "writers"],
        ["hunters", "analysts", "swimmers"],
        ["professors", "divers", "singers"],
        ["models", "directors", "artists"],
        ["riders", "agents", "servers"],
        ["scientists", "philosophers", "artists"],
        ["students", "hikers", "designers"],
        ["assistants", "poets", "scientists"],
        ["lawyers", "workers", "painters"],
        ["doctors", "riders", "investors"],
        ["gamblers", "cleaners", "models"],
        ["guards", "actors", "engineers"],
        ["managers", "administrators", "musicians"],
        ["nurses", "scholars", "buyers"],
        ["riders", "philosophers", "professors"],
        ["divers", "presidents", "cooks"],
        ["surfers", "painters", "senators"],
        ["athletes", "assistants", "chefs"],
        ["administrators", "musicians", "athletes"],
        ["drivers", "senators", "lawyers"],
        ["golfers", "cyclists", "assistants"],
        ["professors", "cooks", "models"],
        ["miners", "teachers", "poets"],
        ["planners", "sailors", "engineers"],
        ["runners", "dentists", "administrators"],
        ["students", "accountants", "doctors"],
        ["therapists", "hikers", "dentists"]]


adjectives = ["happy", "famous", "clever", "tall", "important", "helpful", "popular", "careful", "intelligent", "polite"]


model_name = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

flattened_words = []
for word_triple in word_triples:
    for word in word_triple:
        tokens = tokenizer([" " + word], return_tensors="pt")["input_ids"][0].tolist()
        if len(tokens) > 1:
            print(word)
            print(tokens)
            print("")
        flattened_words.append(word)

flattened_adjectives = []
for adjective in adjectives:
    tokens = tokenizer([" " + adjective], return_tensors="pt")["input_ids"][0].tolist()
    if len(tokens) > 1:
        print(adjective)
        print(tokens)
        print("")
    flattened_adjectives.append(adjective)


flattened_words = list(set(flattened_words))
#print(len(flattened_words))


def gen_example():

    done = False

    while not done:
        lhs = random.choice(list(strongest_syllogism_dict.keys()))
        rhs = strongest_syllogism_dict[lhs]

        x = random.choice(flattened_words)
        if random.choice([True, False]):
            x = random.choice(flattened_adjectives) + " " + x

        y = random.choice(flattened_words)
        if random.choice([True, False]):
            y = random.choice(flattened_adjectives) + " " + y

        z = random.choice(flattened_words)
        if random.choice([True, False]):
            z = random.choice(flattened_adjectives) + " " + z

        full_seq = (x + " " + y + " " + z).split()
        if len(list(set(full_seq))) == len(list(full_seq)):
            done = True

        lhs_orig = lhs[:]
        rhs_orig = rhs[:]

        lhs = lhs.replace("X", x).replace("Y", y).replace("Z", z)
        rhs = rhs.replace("X", x).replace("Y", y).replace("Z", z)

    options = "\\nAll X are Z.\\nSome X are Z.\\nNo X are Z.\\nSome X are not Z.".replace("X", x).replace("Y", y).replace("Z", z)
    query = "We are given two premises:\\n" + lhs + "\\n\\nPlease return the first conclusion listed below that follows from these premises:" + options
    pieces = [query, lhs, rhs, x, y, z, lhs_orig, rhs_orig]
    return "\t".join(pieces)



n_train = 200000
n_valid = 5000
n_test = 5000


examples = []
for _ in range(n_train + n_valid + n_test):
    examples.append(gen_example())

train_set = examples[:n_train]
valid_set = examples[n_train:n_train+n_valid]
test_set = examples[n_train+n_valid:]

fo = open("base_sentences/sentences.train", "w")
fo.write("sentence\ttarget_seq\tanswer\tX\tY\tZ\tLHS\tRHS\n")
for example in train_set:
    fo.write(example + "\n")
fo.close()

fo = open("base_sentences/sentences.valid", "w")
fo.write("sentence\ttarget_seq\tanswer\tX\tY\tZ\tLHS\tRHS\n")
for example in valid_set:
    fo.write(example + "\n")
fo.close()

fo = open("base_sentences/sentences.test", "w")
fo.write("sentence\ttarget_seq\tanswer\tX\tY\tZ\tLHS\tRHS\n")
for example in test_set:
    fo.write(example + "\n")
fo.close()





