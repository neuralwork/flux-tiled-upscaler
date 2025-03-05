from __future__ import annotations
import gc
import math
from collections import namedtuple

import cv2
import numpy as np
from PIL import Image


LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
Grid = namedtuple("Grid", ["tiles", "tile_width", "tile_height", "image_w", "image_h", "overlap"])


def split_grid(image, tile_width=512, tile_height=512, overlap=64):
    """
    Split image to a grid of overlapping tiles.

    Arg:
        image (`PIL.Image`):
            Input Pillow image to split into a grid.
        tile_width (`int`, *optional*, defaults to 512):
            Width of each tile.
        tile_height (`int`, *optional*, defaults to 512):
            Height of each tile.
        overlap (`int`, *optional*, defaults to 64):
            Overlapping pixels between each tile.
    Returns:
        [`Grid`]: A named tuple of resulting tiles.
    """
    w = image.width
    h = image.height

    non_overlap_width = tile_width - overlap
    non_overlap_height = tile_height - overlap

    cols = math.ceil((w - overlap) / non_overlap_width)
    rows = math.ceil((h - overlap) / non_overlap_height)

    dx = (w - tile_width) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_height) / (rows - 1) if rows > 1 else 0

    grid = Grid([], tile_width, tile_height, w, h, overlap)
    for row in range(rows):
        row_images = []

        y = int(row * dy)
        if y + tile_height >= h:
            y = h - tile_height

        for col in range(cols):
            x = int(col * dx)
            if x + tile_width >= w:
                x = w - tile_width

            tile = image.crop((x, y, x + tile_width, y + tile_height))
            row_images.append([x, tile_width, tile])

        grid.tiles.append([y, tile_height, row_images])

    return grid


def combine_grid(grid):
    """
    Stitch upscaled tiles into one image.

    Arg:
        grid (`grid`):
            A named tuple of upscaled tiles.
    Returns:
        [`PIL.Image`]: A stitched image.
    """
    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, 'L')

    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_height, axis=0))
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    return combined_image