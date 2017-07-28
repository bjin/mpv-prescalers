This repo contains scripts that generate user shader for prescaling in
[mpv](https://mpv.io/).

For the generated user shaders, check the [master branch](https://github.com/bjin/mpv-prescalers/tree/master).

# Usage

Python 3 is required. `./gen.sh` will generate all user shaders in
current work directory.

Alternatively, you could generate shaders with customized options:
```
./nnedi3.py --target yuv --nns 32 --win 8x4 --max-downscaling-ratio 1.8 > ~/.config/mpv/shaders/nnedi3.hook
```

Or play video directly with scripts
```
mpv --opengl-shaders=<(path/to/superxbr.py --target=native) video.mkv
```

# About RAVU

RAVU is an experimental prescaler based on RAISR (Rapid and Accurate Image Super
Resolution). It adopts the core idea of RAISR for upscaling, without adopting
any further refinement RAISR used for post-processing, including blending and
sharpening.

RAVU is a convolution kernel based upscaling algorithm. The kernel is trained
from large amount of pictures with a straightforward linear regression model.
From a high level view, it's kind of similar to EWA scalers, but more adaptive
to local gradient features, and probably would result in less aliasing. RAVU is
**NOT** neural network based, it won't create details from thin air, and the
upscaled image tends to be less sharp compared to ground truth.

Currently RAVU is still experimental. There are issues with `radius=4` kernel
due to its large size. The model is also not well tuned, most parameters
are taken from RAISR paper directly without actual evaluation. Initial results
shows that mixing different kind of sources in training set would hurt final
results considerably, so the current model is only tuned for video that I
actually care about. It's not properly trained against all kind of video.

# License

`nnedi3` and `superxbr` were ported from [MPDN
project](https://github.com/zachsaw/MPDN_Extensions), and were originally
licensed under terms of [LGPLv3](https://www.gnu.org/licenses/lgpl-3.0.en.html).
`superxbr` shader in addition was licensed under terms of a more permissive
license ([MIT](https://opensource.org/licenses/MIT)).

The ported shaders (in mpv) also include contributions licensed under terms of
LGPLv2 or later (particularly, major part of `superxbr` was rewritten by
@haasn).

As a whole, shaders in this repo are licensed with LGPLv3.
