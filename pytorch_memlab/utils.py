from calmsize import size as calmsize
from math import isnan

def readable_size(num_bytes):
    return '' if isnan(num_bytes) else '{:.2f}'.format(calmsize(num_bytes))
