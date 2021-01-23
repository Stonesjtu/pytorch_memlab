from math import isnan
from calmsize import size as calmsize

def readable_size(num_bytes: int) -> str:
    return '' if isnan(num_bytes) else '{:.2f}'.format(calmsize(num_bytes))
