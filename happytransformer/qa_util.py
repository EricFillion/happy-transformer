from collections import namedtuple

SumPair = namedtuple('SumPair',['idx1', 'idx2', 'sum'])

def biggest_sums(items_a, items_b):
    '''
    assumes items_a and items_b sorted (descending)
    return (idx1,idx2,sum) for all the biggest sums
    from items_a and items_b, in decreasing order
    '''
    a_index = b_index = 0
    while a_index < len(items_a) and b_index < len(items_b):
        yield SumPair(
            a_index, b_index,
            sum=items_a[a_index] + items_b[b_index]
        )
        # increment in whichever direction has smaller gain
        # fallback to -inf at end of list. 
        # this will always be taken last.
        next_from_a = items_a[a_index+1] if a_index + 1 < len(items_a) else float('-inf')
        next_from_b = items_b[b_index+1] if b_index + 1 < len(items_b) else float('-inf')

        diff_a = items_a[a_index] - next_from_a
        diff_b = items_b[b_index] - next_from_b

        if diff_a >= diff_b:
            b_index += 1
        else:
            a_index += 1