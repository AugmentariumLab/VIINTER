# VIINTER
Implementation of VIINTER: View Interpolation With Implicit Neural Representations of Images

# Quick Start Example
Dependencies: torch 1.12, [clip](https://github.com/openai/CLIP), imageio, tqdm, skimage, argparse, PIL

### Train a Stanford Light Field scene
1. Download ```Chess``` scene [here](http://lightfield.stanford.edu/lfs.html)
2. Move ```rectified``` to ```data/LF``` and rename as ```data/LF/Chess```
3. ```python train.py --data_dir data/LF --dset LF --scene Chess --clip 0.0```

### Train a LLFF scene
1. Download ```flower``` scene [here](https://github.com/bmild/nerf)
2. Move ```flower/images_4``` under  ```data/LLFF``` and rename as ```data/LLFF/flower```
3. ```python train.py --data_dir data/LLFF --dset LLFF --scene flower --clip 0.01```
