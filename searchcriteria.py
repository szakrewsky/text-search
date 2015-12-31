__author__ = 'sz372'


_NUMBERS =[chr(i) for i in range(48,58)]
_UPPER = [chr(i) for i in range(65,91)]
_LOWER = [chr(i) for i in range(97,123)]
_SAMPLES = _NUMBERS + _UPPER + _LOWER
_LABELS = {_SAMPLES[i]: i + 1 for i in range(0, len(_SAMPLES))}


def get_label_value(i):
    return _SAMPLES[i-1]


def _get_label_indices(token):
    if token == '\d':
        return range(1,11)
    elif token in _LABELS:
        return [_LABELS[token.upper()], _LABELS[token.lower()]]
    else:
        return []


class SearchCriteria(object):

    def __init__(self, tokens=[]):
        self.tokens = tokens

    @staticmethod
    def parse(s):
        tokens = []
        for c in s:
            if tokens and tokens[-1] == '\\' and c == 'd':
                tokens[-1] = '\d'
            else:
                tokens.append(c.lower())
        return SearchCriteria(tokens)

    def indexset(self):
        sc_label_indices = set()
        for token in self.tokens:
            sc_label_indices = sc_label_indices.union(_get_label_indices(token))
        return sc_label_indices