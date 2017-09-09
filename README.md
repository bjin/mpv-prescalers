This repo contains scripts that generate user shader for prescaling in
[mpv](https://mpv.io/).

For the generated user shaders, check the [master branch](https://github.com/bjin/mpv-prescalers/tree/master).

# Usage

Python 3 is required. `./gen.sh` will generate all user shaders in
current work directory.

Alternatively, you could generate shader with customized options:
```
./nnedi3.py --target luma --nns 32 --win 8x4 --max-downscaling-ratio 1.8 > ~/.config/mpv/shaders/nnedi3.hook
```

Or play video directly with scripts
```
mpv --opengl-shaders=<(path/to/ravu.py --weights-file path/to/ravu_weights-r3.py --use-gather) video.mkv
```

# About RAVU

RAVU is an experimental prescaler based on RAISR (Rapid and Accurate Image Super
Resolution). It adopts the core idea of RAISR for upscaling, without adopting
any further refinement RAISR used for post-processing, including blending and
sharpening.

RAVU is a convolution kernel based upscaling algorithm. The kernels are trained
from large amount of pictures with a straightforward linear regression model.
From a high level view, it's kind of similar to EWA scalers, but more adaptive
to local gradient features, and would produce lesser aliasing. Like EWA scalers,
currently, plain RAVU would also produce noticeable ringings.

RAVU-Lite is a faster, slightly-lower-quality and luma-only variant of RAVU.

# License

Shaders in this repo are licensed under terms of LGPLv3. Check the header of
each file for details.
