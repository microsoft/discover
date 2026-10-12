
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


for d1 in digits:
    for d2 in digits:
        for d3 in digits:
            for o1, o2 in operations:
                expression = " ".join([d1, o1, d2, o2, d3])
                total = apply_arithmetic(expression.split())
                expressions.append([expression, str(total)])



random.shuffle(expressions)
total = len(expressions)


test = expressions[:2000]
valid = expressions[2000:3000]
train = expressions[3000:]


fo_train = open("base_sentences/sentences.train", "w")
fo_train.write("sentence\ttarget_seq\tanswer\n")
for example in train:
    fo_train.write(example[0] + "\t" + example[0] + "\t" + example[1] + "\n")
fo_train.close()


fo_valid = open("base_sentences/sentences.valid", "w")
fo_valid.write("sentence\ttarget_seq\tanswer\n")
for example in valid:
    fo_valid.write(example[0] + "\t" + example[0] + "\t" + example[1] + "\n")
fo_valid.close()


fo_test = open("base_sentences/sentences.test", "w")
fo_test.write("sentence\ttarget_seq\tanswer\n")
for example in test:
    fo_test.write(example[0] + "\t" + example[0] + "\t" + example[1] + "\n")
fo_test.close()




