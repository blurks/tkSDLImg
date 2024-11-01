"""Tk widget for showing an image using SDL.

Provide an image widget that is capable of frequent refreshs, eg. for
animations or videos.
"""
import itertools
from typing import Optional, Tuple

import cv2

import numpy as np

import tkinter as tk
import tkinter.ttk as ttk

import sdl2  # type: ignore[import-untyped]
import sdl2.ext  # type: ignore[import-untyped]
from sdl2.ext.window import _check_video_init  # type: ignore[import-untyped]
from sdl2.ext.err import raise_sdl_err  # type: ignore[import-untyped]


sdl2.ext.init()


def fit_dimensions(
        src_size: Tuple[int, int], dst_size: Tuple[int, int]
) -> Tuple[int, int]:
    """Scale dimensions preserving the original aspect ratio.

    Scale dimensions described by `src_size` to fit into a rectangle of
    `dst_size`.

    Parameters:
    src_size - width and height of original rectangle
    dst_size - max width and height of scaled rectangle

    Returns:
    dimensions of a new rectangle that fits `dst_size`

    Exceptions:
    ValueError - if width or height of src or dst is <= 0
    """
    if any(map(lambda x: x <= 0, src_size + dst_size)):
        raise ValueError("Dimensions not greater zero")

    src_ratio = src_size[0] / src_size[1]
    dst_ratio = dst_size[0] / dst_size[1]

    if src_ratio > dst_ratio:
        width = dst_size[0]
        factor = dst_size[0] / src_size[0]
        height = int(np.ceil(src_size[1] * factor))
    else:
        height = dst_size[1]
        factor = dst_size[1] / src_size[1]
        width = int(np.ceil(src_size[0] * factor))

    return (width, height)


class SDLWindowFromId(sdl2.ext.Window):
    """SDL window created from existing native window.

    cf. `sdl2.ext.Window`
    """

    def __init__(self, winfo_id: int) -> None:
        """Create a SDL window from existing native window.

        Parameters:
        winfo_id - handle of existing window (eg. `widget.winfo_id()` in tk)
        """
        _check_video_init("creating a window from id")
        self.window = None
        self._winfo_id = winfo_id
        self.create()

    def create(self) -> None:
        """(Re-)Create the SDL window."""
        if self.window:
            return  # type: ignore[unreachable]
        window = sdl2.SDL_CreateWindowFrom(self._winfo_id)
        if not window:
            raise_sdl_err("creating the window from id")
        self.window = window.contents


class ImgWidget():
    """Wrapper around a tk widget showing an image.

    The tkinter widget can be accesed through the `widget` variable.
    """

    _sequence_number = itertools.count(0)

    __slots__ = (
        "_id",
        "orig_img",
        "interpolation",
        "widget",
        "sdl_window",
        "windowsurface",
    )

    def __init__(
            self,
            parent: tk.BaseWidget | tk.Tk,
            img: np.ndarray,
            interpolation: int = cv2.INTER_AREA,
    ) -> None:
        """Create a tkinter widget showing the image represented by `img`.

        Parameters:
        parent - tkinter parent
        img - numpy array representation of the image (eg. as returned by
          `cv2.imread()`)
        interpolation - interpolation method to use when resizing the image (cf.
          `cv2.resize()`)
        """
        self._id = next(type(self)._sequence_number)
        self.orig_img = img
        self.interpolation = interpolation
        self.widget = tk.Frame(parent, background="", name=f"sdl{self._id}")
        self.sdl_window: Optional[sdl2.ext.window.Window] = None
        self.windowsurface: Optional[sdl2.SDL_Surface] = None
        self.widget.bind("<Configure>", self._resize_cb, add=True)
        self.widget.bind("<Expose>", self._expose_cb, add=True)
        self.widget.bind("<Destroy>", self._destroy_cb, add=True)
        self.widget.bind("<Map>", self._map_cb, add=True)
        self.redraw()

    def _init_sdl(self) -> None:
        self.sdl_window = SDLWindowFromId(self.widget.winfo_id())
        self.windowsurface = self.sdl_window.get_surface()

    def update_img(self, img: np.ndarray) -> None:
        """Update the shown image.

        Parameters:
        img - numpy array representation of the image (eg. as returned by
          `cv2.imread()`)
        """
        self.orig_img = img
        self.redraw()

    def redraw(self) -> None:
        """Redraw the image."""
        w_width = self.widget.winfo_width()
        w_height = self.widget.winfo_height()
        self._fit_img(w_width, w_height)

    def _fit_img(self, width: int, height: int) -> None:
        if self.sdl_window is None or width == 0 or height == 0:
            return
        orig_size = self.orig_img.shape[1::-1]
        new_size = fit_dimensions(orig_size, (width, height))
        scaled_img = cv2.resize(
            self.orig_img, new_size, interpolation=self.interpolation)
        bgra_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2BGRA)
        sdl2.ext.fill(self.windowsurface, (0, 0, 0))
        windowArray = sdl2.ext.pixels3d(self.windowsurface, transpose=False)
        x_offset = (width - new_size[0]) // 2
        y_offset = (height - new_size[1]) // 2
        windowArray[
            y_offset:y_offset + new_size[1],
            x_offset:x_offset + new_size[0]
        ] = bgra_img
        self.sdl_window.refresh()

    def _resize_cb(self, event: tk.Event) -> None:
        if self.sdl_window is None:
            return
        self.sdl_window.size = (event.width, event.height)
        self.windowsurface = self.sdl_window.get_surface()
        self._fit_img(event.width, event.height)

    def _destroy_cb(self, event: tk.Event) -> None:
        if self.sdl_window is not None:
            self.sdl_window.close()

    def _expose_cb(self, event: tk.Event) -> None:
        if self.sdl_window is not None:
            self.sdl_window.refresh()

    def _map_cb(self, event: tk.Event) -> None:
        self._init_sdl()
        self.redraw()


def _main() -> None:
    import sys
    if len(sys.argv) != 2:
        print(f"USAGE: {sys.argv[0]} <filename>", file=sys.stderr)
        exit(1)

    try:
        img = cv2.imread(sys.argv[1])
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)

    if img is None:
        print(f"Could not open file at {sys.argv[1]}")  # type: ignore[unreachable]
        exit(1)

    root = tk.Tk()
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frame = ttk.Frame(root)
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)
    frame.grid(column=0, row=0, sticky="nwes")

    img_widget = ImgWidget(frame, img)
    img_widget.widget.grid(column=0, row=0, sticky="nwes")

    def _quit_cb(event: tk.Event) -> None:
        root.destroy()

    root.bind_all("<KeyPress-q>", _quit_cb)

    root.mainloop()
    exit(0)


if __name__ == "__main__":
    _main()
