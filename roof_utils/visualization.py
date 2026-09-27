from PIL import Image, ImageDraw
import numpy as np

def show_row(rgb, roof, valid, title):
    yy, xx = np.indices(roof.shape)
    checker = np.where((yy // 12 + xx // 12) % 2, 200, 235).astype(np.uint8)
    visible = rgb.copy()
    visible[valid == 0] = np.repeat(checker[..., None], 3, axis=2)[valid == 0]
    overlay = visible.copy()
    marked = (roof == 1) & (valid == 1)
    overlay[marked] = (0.55 * visible[marked] + 0.45 * np.array([255, 30, 30])).astype(np.uint8)
    row = Image.new('RGB', (1024, 302), 'white')
    draw = ImageDraw.Draw(row)
    draw.text((5, 4), title, fill='black')
    for j, (label, a) in enumerate([('RGB', visible), ('Roof overlay', overlay), ('Binary roof', roof * 255), ('Validity', valid * 255)]):
        draw.text((j * 256 + 5, 25), label, fill='black')
        row.paste(Image.fromarray(a).convert('RGB'), (j * 256, 46))
    return row
