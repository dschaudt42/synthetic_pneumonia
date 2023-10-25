import glob
import pandas as pd
from pathlib import Path

folder = '../../data/segmentation_test/train_img_captioning'
img_paths = glob.glob(f'{folder}/data/*')
img_names = [Path(x).name for x in img_paths]

df = pd.DataFrame(img_names,columns=['file_name'])

df.loc[df['file_name'].str.startswith('C'),'text'] = "An x-ray image of the lung with covid-19 pneumonia"
df.loc[df['file_name'].str.startswith('B'),'text'] = "An x-ray image of the lung with bacterial pneumonia"
df.loc[df['file_name'].str.startswith('NB'),'text'] = "An x-ray image of the lung, healthy patient, no signs of pneumonia"
df.loc[df['file_name'].str.startswith('P'),'text'] = "An x-ray image of the lung with fungal pneumonia"
df.loc[df['file_name'].str.startswith('V'),'text'] = "An x-ray image of the lung with viral pneumonia"

#df['file_name'] = 'data/' + df['image'].astype(str)

df.to_json('../../data/segmentation_test/train_img_captioning/metadata.jsonl', orient='records', lines=True)

#out = df.to_json(orient='records')[1:-1]

#with open('../../data/segmentation_test/train_img_captioning/metadata.json1','w') as f:
#    f.write(out)