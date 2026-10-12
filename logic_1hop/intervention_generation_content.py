

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

def gen_example():

    done = False

    while not done:
        lhs = random.choice(list(strongest_syllogism_dict.keys()))
        rhs = strongest_syllogism_dict[lhs]

        x = random.choice(flattened_words)
        if random.choice([True, False]):
            x = random.choice(flattened_adjectives) + " " + x

        new_adj = random.choice(flattened_adjectives)

        y = random.choice(flattened_words)
        if random.choice([True, False]):
            y = random.choice(flattened_adjectives) + " " + y

        z = random.choice(flattened_words)
        if random.choice([True, False]):
            z = random.choice(flattened_adjectives) + " " + z

        replace_options = x + " " + z
        replace_options = replace_options.split()
        to_replace = random.choice(replace_options)
        if to_replace in flattened_adjectives:
            new_word = random.choice(flattened_adjectives)
        else:
            new_word = random.choice(flattened_words)

        full_seq = (x + " " + y + " " + z + " " + new_word).split()
        if len(list(set(full_seq))) == len(list(full_seq)):
            done = True

        lhs_orig = lhs[:]
        rhs_orig = rhs[:]

        lhs = lhs.replace("X", x).replace("Y", y).replace("Z", z)
        rhs = rhs.replace("X", x).replace("Y", y).replace("Z", z)

    fr_pairs_old = "(DUMMY," + to_replace + ")"
    fr_pairs_new = "(DUMMY," + new_word + ")"

    answer_new = rhs.replace(to_replace, new_word)

    options = "\\nAll X are Z.\\nSome X are Z.\\nNo X are Z.\\nSome X are not Z.".replace("X", x).replace("Y", y).replace("Z", z)
    query = "We are given two premises:\\n" + lhs + "\\n\\nPlease return the first conclusion listed below that follows from these premises:" + options
    pieces = [query, lhs, rhs, x, y, z, lhs_orig, rhs_orig, answer_new, fr_pairs_old, fr_pairs_new]

    pieces2 = [query.replace(to_replace, new_word), lhs.replace(to_replace, new_word), rhs.replace(to_replace, new_word)]
    return "\t".join(pieces), "\t".join(pieces2)


n_interventions = 100


examples = []
for _ in range(n_interventions):
    examples.append(gen_example())


fo = open("base_sentences/interventions.content", "w")
fo_eval = open("base_sentences/interventions_eval.content", "w")

fo_eval.write("sentence\ttarget_seq\tanswer\n")
fo.write("sentence\ttarget_seq\tanswer\tX\tY\tZ\tLHS\tRHS\tnew_answer\tfr_pairs_old\tfr_pairs_new\n")
for example, example_eval in examples:
    fo.write(example + "\n")
    fo_eval.write(example_eval + "\n")
fo.close()
fo_eval.close()








