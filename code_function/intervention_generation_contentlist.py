


import random





def suffix(text, ending):

    text = text + [ending]

    return text


def prefix(text, start):

    text = [start] + text

    return text


def repeat(text, count):

    text = text * count

    return text


def reverse(text):

    return text[::-1]

def sort_list(text):
    return sorted(text)



alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def gen_full():

    found_unique = False
    while not found_unique:

        x = ""
        for _ in range(random.choice([3])):
            x += random.choice(list(alphabet))

        y = ""
        for _ in range(random.choice([3])):
            y += random.choice(list(alphabet))

        x = list(x)
        y = list(y)

        combined = x + y
        if len(list(set(combined))) == len(combined):
            found_unique = True

    ordering = [("x", x), ("y", y)]
    random.shuffle(ordering)

    lines = []
    for name, value in ordering:
        list_string = '", "'.join([char for char in value])
        lines.append(name + ' = ["' + list_string + '"]\\n')

    lines = "".join(lines)

    ordering_args = [("x", x), ("y", y)]
    random.shuffle(ordering_args)

    arg_name = ordering_args[0][0]
    base_string = ordering_args[0][1]
    replacement_index = random.choice(list(range(len(base_string))))
    letter_to_replace = base_string[replacement_index]

    rep_count = random.choice([2, 3, 4])


    found_unique_extras = False
    while not found_unique_extras:
        suffix_letter = random.choice(list(alphabet))
        prefix_letter = random.choice(list(alphabet))
        mod_letter = random.choice(list(alphabet))
        replacement_letter = random.choice(list(alphabet))

        new_letters = [suffix_letter, prefix_letter, mod_letter, replacement_letter]
        combined = x + y + new_letters
        if len(list(set(combined))) == len(combined):
            found_unique_extras = True

    f_types = [
            ("reverse", lambda x: reverse(x), "def NAME(sequence):\\n\\n\\treturn sequence[::-1]\\n\\n"),
            ("sort", lambda x: sort_list(x), "def NAME(sequence):\\n\\n\\treturn sorted(sequence)\\n\\n"),
            ("repeat_" + str(rep_count), lambda x: repeat(x, rep_count), "def NAME(sequence):\\n\\n\\treturn sequence * " + str(rep_count) + "\\n\\n"),
            ("suffix_" + suffix_letter, lambda x: suffix(x, suffix_letter), "def NAME(sequence):\\n\\n\\treturn sequence + [\"" + str(suffix_letter) + "\"]\\n\\n"),
            ("prefix_" + prefix_letter, lambda x: prefix(x, prefix_letter), "def NAME(sequence):\\n\\n\\treturn [\"" + str(prefix_letter) + "\"] + sequence\\n\\n"),
            ]

    modified_base_string = base_string[:replacement_index] + [replacement_letter] + base_string[replacement_index+1:]

    lines = []
    for name, value in ordering:
        list_string = '", "'.join([char for char in value])
        lines.append(name + ' = ["' + list_string + '"]\\n')

    lines = "".join(lines)


    lines_modified = lines.replace(letter_to_replace, replacement_letter)


    random.shuffle(f_types)
    change = f_types[0]
    alter = f_types[1]

    f_type_vars = [("change", change), ("alter", alter)]
    random.shuffle(f_type_vars)
    f_type_lines = []
    for name, value in f_type_vars:
        f_type_line = value[2].replace("NAME", name)
        f_type_lines.append(f_type_line)
    f_type_lines = "".join(f_type_lines)


    f_type = random.choice(f_type_vars)
    f_name = f_type[0]
    f_category = f_type[1][0]
    f_funct = f_type[1][1]

    modified_string = f_funct(base_string)
    modified_modified_string = f_funct(modified_base_string)
    command = f_name + "(" + arg_name + ")"

    full_command = f_type_lines + lines + command
    full_command_modified = f_type_lines + lines_modified + command
    order = [elt[0] for elt in ordering_args]
    order = "".join(order)


    modified_x = list("".join(x).replace(letter_to_replace, replacement_letter))
    modified_y = list("".join(y).replace(letter_to_replace, replacement_letter))

    return full_command, change[0], alter[0], "".join(x), "".join(y), order, "".join(modified_string), modified_string, modified_modified_string, "(DUMMY," + letter_to_replace + ")", "(DUMMY," + replacement_letter + ")", full_command_modified, modified_x, modified_y



def example_and_function():

    done = False
    while not done:
        query, change, alter, x, y, order, answer_str, answer, new_answer, fr_pairs_old, fr_pairs_new, full_command_modified, modified_x, modified_y = gen_full()
        done = True

    return query, change, alter, x, y, order, answer_str, answer, new_answer, fr_pairs_old, fr_pairs_new, full_command_modified, modified_x, modified_y


n_intervention = 100


examples = []
examples_intervention = []
for _ in range(n_intervention):
    query, change, alter, x, y, order, answer_str, answer, new_answer, fr_pairs_old, fr_pairs_new, full_command_modified, modified_x, modified_y = example_and_function()
    example = full_command_modified + "\t" + full_command_modified + "\t" + change + "\t" + alter + "\t" + str(modified_x) + "\t" + str(modified_y) + "\t" + order + "\t" + "".join(new_answer) + "\t" + str(new_answer) + "\n"
    example = example.replace("'", '"')

    example_intervention = query + "\t" + query + "\t" + change + "\t" + alter + "\t" + str(x) + "\t" + str(y) + "\t" + order + "\t" +  str(answer_str) + "\t" + str(answer) + "\t" + str(new_answer) + "\t" + fr_pairs_old + "\t" + fr_pairs_new + "\n"
    example_intervention = example_intervention.replace("'", '"')

    examples.append(example)
    examples_intervention.append(example_intervention)


fo = open("base_sentences/interventions.contentlist", "w")
fo_eval = open("base_sentences/interventions_eval.contentlist", "w")
fo_eval.write("sentence\ttarget_seq\tchange\talter\tx\ty\torder\tanswer_str\tanswer\n")
fo.write("sentence\ttarget_seq\tchange\talter\tx\ty\torder\tanswer_str\tanswer\tnew_answer\tfr_pairs_old\tfr_pairs_new\n")
for example, example_intervention in zip(examples, examples_intervention):
    fo_eval.write(example)
    fo.write(example_intervention)





