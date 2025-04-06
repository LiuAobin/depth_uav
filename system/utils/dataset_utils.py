from PIL import Image


def read_lines(filename):
    """
    读取文件所有行，并以列表返回
    Args:
        filename: 文件名
    Returns:
    """
    with open(filename,'r') as f:
        lines = f.read().splitlines()
    return lines

def pil_loader(img_path):
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')