from PIL import Image
import glob
from pathlib import Path
from tqdm import tqdm

input_images = glob.glob('/data/DS/Projekte/covid19/data/segmentation_test/train/*.png')
output_folder = '/data/DS/Projekte/covid19/data/segmentation_test/train_256'

for image in tqdm(input_images):
    im = Image.open(image).resize((256,256))
    im.save(f'{output_folder}/{Path(image).name}')