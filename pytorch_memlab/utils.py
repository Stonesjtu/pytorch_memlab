from calmsize import size as calmsize


def readable_size(num_bytes):
    return '{:.2f}'.format(calmsize(num_bytes))
