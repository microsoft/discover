

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import random


digits = [str(x) for x in [elt - 10 for elt in list(range(1,20))]]
operations = [("*", "+"), ("+", "*")]

expressions = []


def apply_arithmetic(expr_orig):

    expr_string = " ".join(expr_orig)
    expr = expr_string.split()

    if expr[-1] == "=":
        return apply_arithmetic(expr[:-1])

    if expr[-1] == "*" or expr[-1] == "+" or expr[-1] == "-":
        return apply_arithmetic(expr[:-1])

    if "(" in expr:
        if ")" in expr:
            ind_open = expr.index("(")
            ind_close = expr.index(")")
            sub_expr = expr[ind_open+1:ind_close]
            sub_expr_value = apply_arithmetic(sub_expr)
            new_expr = expr[:ind_open] + [str(sub_expr_value)] + expr[ind_close+1:]
            return apply_arithmetic(new_expr)
        else:
            return apply_arithmetic(expr_string.replace("( ", "").split())

    if "*" in expr:
        index_times = expr.index("*")
        n1 = expr[index_times-1]
        n2 = expr[index_times+1]

        new_expr = expr[:index_times-1] + [str(int(n1)*int(n2))] + expr[index_times+2:]

        return apply_arithmetic(new_expr)


    if "+" in expr:
        index_plus = expr.index("+")
        n1 = expr[index_plus-1]
        n2 = expr[index_plus+1]

        new_expr = expr[:index_plus-1] + [str(int(n1)+int(n2))] + expr[index_plus+2:]

        return apply_arithmetic(new_expr)

    if "-" in expr:
        index_minus = expr.index("-")
        n1 = expr[index_minus-1]
        n2 = expr[index_minus+1]

        new_expr = expr[:index_minus-1] + [str(int(n1)-int(n2))] + expr[index_minus+2:]

        return apply_arithmetic(new_expr)

    if len(expr) == 1:
        return int(expr[0])

n_interventions = 100
interventions = []
for _ in range(n_interventions):
    found = False
    while not found:
        d1 = random.choice(digits)
        d2 = random.choice(digits)
        d3 = random.choice(digits)

        selected_digits = [abs(int(d1)), abs(int(d2)), abs(int(d3))]
        if len(list(set(selected_digits))) == len(selected_digits) and d1 != "0" and d2 != "0" and d3 != "0":
            found = True

    o1, o2 = random.choice(operations)

    expression = " ".join([d1, o1, d2, o2, d3])
    total = apply_arithmetic(expression.split())


    if random.choice([True, False]):
        found = False
        while not found:
            new_d1 = random.choice(digits)
            all_selected_digits = [abs(int(d1)), abs(int(d2)), abs(int(d3)), abs(int(new_d1))]
            if len(list(set(all_selected_digits))) == len(all_selected_digits):
                if int(new_d1) < 0 and int(d1) < 0:
                    found = True
                elif int(new_d1) >= 0 and int(d1) >= 0:
                    found = True

            new_expression = " ".join([new_d1, o1, d2, o2, d3])
            new_total = apply_arithmetic(new_expression.split())

            if new_total == total:
                found = False

        fr_pairs_old = "(DUMMY," + d1.replace("-", "") + ")"
        fr_pairs_new = "(DUMMY," + new_d1.replace("-", "") + ")"
    else:
        found = False
        while not found:
            new_d2 = random.choice(digits)
            all_selected_digits = [abs(int(d1)), abs(int(d2)), abs(int(d3)), abs(int(new_d2))]
            if len(list(set(all_selected_digits))) == len(all_selected_digits):
                if int(new_d2) < 0 and int(d2) < 0:
                    found = True
                elif int(new_d2) >= 0 and int(d2) >= 0:
                    found = True

            new_expression = " ".join([d1, o1, new_d2, o2, d3])
            new_total = apply_arithmetic(new_expression.split())
            if new_total == total:
                found = False

        fr_pairs_old = "(DUMMY," + d2.replace("-", "") + ")"
        fr_pairs_new = "(DUMMY," + new_d2.replace("-", "") + ")"

    interventions.append([expression, str(total), new_expression, str(new_total), fr_pairs_old, fr_pairs_new])


fo = open("base_sentences/interventions.content", "w")
fo_eval = open("base_sentences/interventions_eval.content", "w")
fo.write("sentence\ttarget_seq\tanswer\tnew_answer\tfr_pairs_old\tfr_pairs_new\n")
fo_eval.write("sentence\ttarget_seq\tanswer\n")
for example in interventions:
    fo.write(example[0] + "\t" + example[0] + "\t" + example[1] + "\t" + example[3] + "\t" + example[4] + "\t" + example[5] + "\n")
    fo_eval.write(example[2] + "\t" + example[2] + "\t" + example[3] + "\n")
fo.close()
fo_eval.close()





