"""Regenerate the tiny, synthetic EXIF fixture; never uses dataset photos."""

from pathlib import Path

from PIL import Image, ImageDraw


image = Image.new("RGB", (32, 16), "red")
ImageDraw.Draw(image).rectangle((16, 0, 31, 15), fill="blue")
exif = Image.Exif()
exif[274] = 6  # Rotate 90 degrees clockwise: decoded image is 16 x 32.
image.save(Path(__file__).with_name("rotated-colors.jpg"), quality=100, subsampling=0, exif=exif)
