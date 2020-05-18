# -*- coding: utf-8 -*-
abc = "0123456789ABCEHKMOPTXY"
mapping = {
    'А': 'A',
    'В': 'B',
    'С': 'C',
    'Е': 'E',
    'Н': 'H',
    'К': 'K',
    'М': 'M',
    'О': 'O',
    'Р': 'P',
    'Т': 'T',
    'Х': 'X',
    'У': 'Y',
}


def labels_to_text(labels, abc=abc):
    return ''.join(list(map(lambda x: abc[int(x) - 1], labels)))


def text_to_labels(text, abc=abc):
    return list(map(lambda x: abc.index(x) + 1, text))


def is_valid_str(s, abc=abc):
    for ch in s:
        if ch not in abc:
            return False
    return True


def convert_to_eng(text, mapping=mapping):
    return ''.join([mapping.get(a, a) for a in text])



