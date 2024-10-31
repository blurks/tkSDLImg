import pytest
from typing import Tuple

import tkSDLImg


@pytest.mark.parametrize(
    "src_size, dst_size",
    [
        ((0, 0), (0, 0)),
        ((0, 1), (1, 1)),
        ((1, 0), (1, 1)),
        ((1, 1), (0, 1)),
        ((1, 1), (1, 0)),
        ((-1, 1), (1, 1)),
    ]
)
def test_fit_dimensions_failing(
        src_size: Tuple[int, int], dst_size: Tuple[int, int]
) -> None:
    with pytest.raises(ValueError):
        tkSDLImg.fit_dimensions(src_size, dst_size)


@pytest.mark.parametrize(
    "src_size, dst_size, expected",
    [
        ((1, 1), (1, 1), (1, 1)),
        ((2, 1), (1, 1), (1, 1)),
        ((1, 1), (2, 1), (1, 1)),
        ((4, 2), (2, 2), (2, 1)),
        ((2, 4), (2, 2), (1, 2)),
        ((3, 3), (2, 2), (2, 2)),
        ((2, 2), (3, 3), (3, 3)),
        ((1000, 1000), (1001, 1001), (1001, 1001)),
        ((1001, 1001), (1000, 1000), (1000, 1000)),
    ]
)
def test_fit_dimenstions(
        src_size: Tuple[int, int],
        dst_size: Tuple[int, int],
        expected: Tuple[int, int],
) -> None:
    assert tkSDLImg.fit_dimensions(src_size, dst_size) == expected
