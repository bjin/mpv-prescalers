This repo contains user shaders for prescaling in [mpv](https://mpv.io/).

For the scripts generating these user shaders, check the [source
branch](https://github.com/bjin/mpv-prescalers/tree/source).

Shaders in `gather/` [directory](https://github.com/bjin/mpv-prescalers/tree/master/gather)
use `textureGatherOffset` for fast texel lookup. They are generally faster
(especially for RAVU, could bring up to 50% performance boost) but requires `OpenGL 4.0`.
In addition, some driver has buggy implementation of `textureGatherOffset`.
Use these shaders only if they actually work (i.e. no blue screen and no noticeable distortion).

# Filenames

Suffix in the filename indicates the type of planes shader is hooked on
(`ravu` supports only luma plane):

* Without any suffix: Triggered on luma plane only (like `prescale-luma=...` option in `mpv`).
* `-chroma`: Triggered on chroma plane only.
* `-yuv`: Triggered on both luma and chroma planes.
* `-all`: Triggered on all source planes (including non-YUV formats).

For `nnedi3` prescaler, `neurons` and `window` settings are indicated in the
filename.

For `ravu` prescaler, `radius` settings are indicated in the filename. Note
that evaluation results shows that improvement by `radius=4` is minimal (less
than `0.1dB` in PSNR), `radius=3` should be enough for daily purpose.

In addition, `{superxbr,ravu*}-native.hook` are native implementations of
`superxbr` and `ravu`, that will do the upscaling on RGB, and is most likely the one
you want use. `{superxbr,ravu*}-native-yuv.hook` are similar shaders but require
the original source to be YUV.

For example:
* `nnedi3-nns32-win8x4.hook`: user shader for luma `nnedi3` prescaling with `32`
  neurons and a local sampling window size of `8x4`.
* `superxbr-chroma.hook`: user shader for chroma-only `superxbr` prescaling.
* `nnedi3-nns128-win8x6-all.hook`: user shader for `nnedi3` prescaling on all
   planes with `128` neurons and a local sampling window size of `8x6`.

# Usage

You only need to download shaders you actually use. The following part of this
section assumes that they are in `shaders` directory in the `mpv` configure
folder (usually `~/.config/mpv/shaders` on Linux).

Use `opengl-shaders="prescaler.hook"` option to load those shaders.

```
opengl-shaders="~~/shaders/nnedi3-nns32-win8x4.hook"
```

All shaders are for one pass only. If you want to have `4x` upscaling, trigger
the same shader twice. For `4x` luma prescaling:

```
opengl-shaders="~~/shaders/nnedi3-nns32-win8x4.hook,~~/shaders/nnedi3-nns32-win8x4.hook"
```

Pay attention that for `4:2:0` sub-sampled `YUV` video, you need an additional
chroma-only prescaling pass to match the post-prescaling size of luma and
chroma planes (they are still not aligned though):

```
opengl-shaders="~~/shaders/nnedi3-nns32-win8x4-chroma.hook"
opengl-shaders="~~/shaders/nnedi3-nns32-win8x4-yuv.hook,~~/shaders/nnedi3-nns32-win8x4-chroma.hook"
```

# Known Issue

* `nnedi3-nns32-win8x6-{chroma,yuv,all}.hook` are extremely slow with certain
  version of nvidia driver.
* Some setup (macOS with AMD driver, or Mesa with old Intel card) are known to
  have buggy `textureGatherOffset` implementation, which might break `nnedi3`
  shaders.

# About RAVU

RAVU is an experimental prescaler based on RAISR (Rapid and Accurate Image Super
Resolution). It adopts the core idea of RAISR for upscaling, without adopting
any further refinement RAISR used for post-processing, including blending and
sharpening.

RAVU is a convolution kernel based upscaling algorithm. The kernels are trained
from large amount of pictures with a straightforward linear regression model.
From a high level view, it's kind of similar to EWA scalers, but more adaptive
to local gradient features, and probably would produce lesser aliasing. RAVU is
**NOT** neural network based, it won't create details from thin air, and the
upscaled image tends to have less sharpness compare to ground truth.

Currently RAVU is still experimental. There are issues with `radius=4` kernel
due to its large size. The model is also not well tuned, most parameters
are taken from the RAISR paper directly without actual evaluation. Initial results
shows that mixing different kind of sources in training set would hurt final
results considerably, so the current model is only tuned for videos that I
actually care about. It's not properly trained against all kinds of video.

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
