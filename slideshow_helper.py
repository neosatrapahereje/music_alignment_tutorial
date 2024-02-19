import glob
import io
import os
import zipfile
from typing import Iterable, Union
from urllib.request import urlopen

import ipywidgets as wg
import requests
from IPython.display import Image, display
from partitura.utils.misc import PathLike
from PIL.Image import Image as PILImage

try:
    import google.colab

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


if IN_COLAB:
    r = requests.get(
        "https://raw.githubusercontent.com/CPJKU/partitura_tutorial/"
        "main/notebooks/02_alignment/figures/dtw_example_png.zip",
        stream=True,
    )
    archive = zipfile.ZipFile(io.BytesIO(r.content), "r")

else:

    archive = zipfile.ZipFile(os.path.join("figures", "dtw_example_png.zip"), "r")

PNG_DTW_EXAMPLE = [
    Image(io.BytesIO(archive.read(f"dtw_example_{i:02d}.png")).getvalue())
    for i in range(30)
]


def slideshow(image_list: Iterable[Union[PathLike, PILImage]]) -> None:
    """
    An interactive widget to display a slideshow in a Jupyter notebook

    Parameters
    ----------
    image_list : Iterable[PathLike]
        List of images to show in the slideshow.
    """

    if isinstance(image_list[0], str):

        def show_image(slider_val: int) -> Image:
            return Image(image_list[slider_val])

    else:

        def show_image(slider_val: int) -> None:
            return image_list[slider_val]

    wg.interact(
        show_image,
        slider_val=wg.IntSlider(
            min=0,
            max=len(image_list) - 1,
            step=1,
        ),
    )


def dtw_example(interactive: bool) -> None:

    gif_path = "https://raw.githubusercontent.com/CPJKU/partitura_tutorial/main/notebooks/02_alignment/figures/dtw_example.gif"
    gif_data = urlopen(gif_path)
    if interactive:
        slideshow(PNG_DTW_EXAMPLE)
    else:
        display(Image(data=gif_data.read(), format="png"))
