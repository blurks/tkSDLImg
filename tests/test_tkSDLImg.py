import numpy as np
import pytest
import tkinter as tk
from typing import Generator, Tuple

import tkSDLImg


@pytest.fixture
def tk_root() -> Generator[tk.Tk, None, None]:
    root = tk.Tk()
    yield root
    root.destroy()


@pytest.fixture
def test_img() -> np.ndarray:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[8:, :, 2] = 255
    return img


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
        ((1000, 1), (1, 1), (1, 1)),
        ((1000, 2), (2, 1), (2, 1)),
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


def test_widget_ids(tk_root: tk.Tk, test_img: np.ndarray) -> None:
    img_w1 = tkSDLImg.ImgWidget(tk_root, test_img)
    img_w2 = tkSDLImg.ImgWidget(tk_root, test_img)
    assert img_w1.widget.winfo_name() == "sdl0"
    assert img_w2.widget.winfo_name() == "sdl1"
