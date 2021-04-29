import numpy as np


def pred_to_string(pred, abc):
    seq = []
    for i in range(len(pred)):
        label = np.argmax(pred[i])
        seq.append(label - 1)
    out = []
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != -1:
                out.append(seq[i])
        else:
            if seq[i] != -1 and seq[i] != seq[i - 1]:
                out.append(seq[i])
    out = ''.join([abc[c] for c in out])
    return out


def decode_sequence(pred, abc):
    pred = pred.permute(1, 0, 2).cpu().data.numpy()
    outputs = []
    for i in range(len(pred)):
        outputs.append(pred_to_string(pred[i], abc))
    return outputs


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
