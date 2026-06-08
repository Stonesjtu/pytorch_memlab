from math import isnan
from calmsize import size as calmsize

def readable_size(num_bytes: int) -> str:
    try:
        if isnan(num_bytes):
            return ''
    except TypeError:
        pass

    size = calmsize(num_bytes)
    try:
        return '{:.2f}'.format(size)
    except TypeError:
        return str(size)
