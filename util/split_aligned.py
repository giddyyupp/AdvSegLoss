import os
from PIL import Image
import PIL

main_dir = '/media/test/Samhi/GANILLA/fpn-gan/dataset/korky_sketch'
split_dir = '/media/test/Samhi/GANILLA/fpn-gan/dataset/korky_sketch_cycle'

for tt in ['train', 'test']:
    ffs = os.listdir(os.path.join(main_dir, tt))
    for ff in ffs:

        AB_path = os.path.join(main_dir, tt, ff)
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        if not os.path.exists(os.path.join(split_dir, tt+'A')):
            os.makedirs(os.path.join(split_dir, tt+'A'))
        if not os.path.exists(os.path.join(split_dir, tt+'B')):
            os.makedirs(os.path.join(split_dir, tt+'B'))
        a_save_name = os.path.join(split_dir, tt+'A', ff)
        b_save_name = os.path.join(split_dir, tt+'B', ff)
        A.save(a_save_name)
        B.save(b_save_name)