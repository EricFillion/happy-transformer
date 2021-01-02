"""
Contains tests for functions found within qa_util.py
"""

from happytransformer.runners.runner_util import SumPair, biggest_sums
def test_biggest_sums():
    """
    Tests the biggest_sums function
    """
    items_a = [7, 4, 3]
    items_b = [7, 6, 4]

    expected_pairs = [
        SumPair(idx1=0, idx2=0, sum=14),  # 7+7
        SumPair(idx1=0, idx2=1, sum=13),  # 7+6
        SumPair(idx1=0, idx2=2, sum=11),  # 7+4
        SumPair(idx1=1, idx2=2, sum=8),  # 4+4
        SumPair(idx1=2, idx2=2, sum=7)  # 3+4
    ]
    computed_pairs = biggest_sums(items_a, items_b)
    assert all(
        expected_pair == computed_pair
        for expected_pair, computed_pair in 
        zip(expected_pairs, computed_pairs)
    )
