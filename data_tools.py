import json
import copy
import re


def out_expression_list(test, dataloader, num_list, num_stack=None):
    ch = "abcdefghijklmnopqrstuvwsyz"
    max_index = len(dataloader.decode_classes_list)
    res = []
    for i in test:
        # if i == 0:
        #     return res
        if i < max_index - 1:
            idx = dataloader.decode_classes_list[i]
            if idx[0] == "t":
                if ch.index(idx[-1]) >= len(num_list):
                    return None
                res.append(num_list[ch.index(idx[-1])])
            else:
                res.append(idx)
        else:
            pos_list = num_stack.pop()
            c = num_list[pos_list[0]]
            res.append(c)
    return res


def read_json_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def split_data(data):
    """
    划分训练集、验证集和测试集
    :param data:
    :return:
    """
    # t_path = "./data/id_ans_test"
    # v_path = "./data/valid_ids.json"
    # valid_id = read_json_data(v_path)
    # test_id = []
    # with open(t_path, 'r') as f:
    #     for line in f:
    #         test_id.append(line.strip().split('\t')[0])
    train, test, valid = [], [], []
    fold_size1 = int(len(data) * 0.8)
    fold_size2 = int(len(data) * 0.9)
    # valid = data[fold_size1:fold_size2]
    # test = data[fold_size2:]
    # train = data[:fold_size1]
    for key, value in data.items():
        if int(key) <= fold_size1:
            train.append((key, value))
        elif int(key) <= fold_size2:
            valid.append((key, value))
        else:
            test.append((key, value))
    return train, valid, test


def string_2_idx_sen(sen,  vocab_dict):
    """
    返回句子的onehot
    """
    word2id = []
    for word in sen:
        if word in vocab_dict:
            word2id.append(vocab_dict[word])
        elif word == 'PI':
            word2id.append(vocab_dict['3.14'])
    return word2id


def pad_sen(sen_idx_list, max_len=115, pad_idx=0):
    return sen_idx_list + [pad_idx] * (max_len - len(sen_idx_list))


def post_solver(post_equ):
    """
    计算后缀表达式的值
    :param post_equ:
    :return:
    """
    stack = []
    op_list = ['+', '-', '/', '*', '^']
    for elem in post_equ:
        if elem not in op_list:
            op_v = elem
            if '%' in op_v:
                op_v = float(op_v[:-1])/100.0
            stack.append(str(op_v))
        elif elem in op_list:
            op_v_1 = stack.pop()
            op_v_1 = float(op_v_1)
            op_v_2 = stack.pop()
            op_v_2 = float(op_v_2)
            if elem == '+':
                stack.append(str(op_v_2+op_v_1))
            elif elem == '-':
                stack.append(str(op_v_2-op_v_1))
            elif elem == '*':
                stack.append(str(op_v_2*op_v_1))
            elif elem == '/':
                stack.append(str(op_v_2/op_v_1))
            else:
                stack.append(str(op_v_2**op_v_1))
    return stack.pop()


def inverse_temp_to_num(equ_list, num_list):
    """
    将表达式中的temp转换回数字
    :param equ_list:
    :param num_list:
    :return:
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    new_equ_list = []
    for elem in equ_list:
        if 'temp' in elem:
            index = alphabet.index(elem[-1])
            try:
                new_equ_list.append(num_list[index])
            except:
                return []
        elif 'PI' == elem:
            new_equ_list.append('3.14')
        else:
            new_equ_list.append(elem)
    return new_equ_list


def compute_prefix_expression(pre_fix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    pre_fix = copy.deepcopy(pre_fix)
    pre_fix.reverse()
    for p in pre_fix:
        if p not in operators:
            pos = re.search("\d+\(", p)
            if pos:
                st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
            elif p[-1] == "%":
                st.append(float(p[:-1]) / 100)
            else:
                st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if b == 0:
                return None
            st.append(a / b)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a - b)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
                return None
            st.append(a ** b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None

def from_infix_to_prefix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = copy.deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            try:
                c = st.pop()
                while c != ")":
                    res.append(c)
                    c = st.pop()
            except IndexError as err:
                err=0
        elif e == "[":
            try:
                c = st.pop()
                while c != "]":
                    res.append(c)
                    c = st.pop()
            except IndexError as err:
                err=0
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res