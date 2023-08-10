This repo contains user shaders for prescaling in [mpv](https://mpv.io/).

For the scripts generating these user shaders, check the [source
branch](https://github.com/bjin/mpv-prescalers/tree/source).

Shaders in [`gather/` directory](https://github.com/bjin/mpv-prescalers/tree/master/gather)
and [`compute/` directory](https://github.com/bjin/mpv-prescalers/tree/master/compute)
are **generally faster**, and are thus recommended to try and use first.

* `gather/` shaders uses `textureGather()` functions, and are usually faster for luma upscalers (`nnedi3`, `ravu`, `ravu-lite` and `ravu-zoom`).
* `compute/` shaders are compute shaders, and are faster for `-yuv` and `-rgb` upscalers.

If you are using an old graphics card (or somehow broken GPU driver), and shaders in none of above directories works. Shaders in root folder
are safe fallback option with minimal requirement for GPU driver or OpenGL version.

If you are using `--vo=gpu` along with `--gpu-api=d3d11`
and encountered the following error:

```
[vo/gpu] Unrecognized/unavailable FORMAT name: 'rgba16f'!
```

You can either switch to `--vo=gpu-next` (libplacebo required) or find shaders
with `rgba16hf` format inside the `vulkan/` folder [here](https://github.com/bjin/mpv-prescalers/tree/b3ed4322cd24b534e7ccc4d4727fced2dfc57c6e/vulkan).

# Usage

You only need to download shaders you actually use. The following part of this
section assumes that they are in `shaders` directory in the `mpv` configure
folder (usually `~/.config/mpv/shaders` on Linux).

Use `glsl-shaders="prescaler.hook"` option to load those shaders. (This will
override other user shaders, use `glsl-shaders-append` in that case)

```
glsl-shaders="~~/shaders/ravu-lite-r3.hook"
```

All shaders are for one pass only. If you want to have `4x` upscaling, trigger
the same shader twice. All the shaders here are generated with
`max-downscaling-ratio` set to `1.414213`. They will be disabled if upscaling is not necessary.

```
glsl-shaders-append="~~/shaders/ravu-lite-r3.hook"
glsl-shaders-append="~~/shaders/ravu-lite-r3.hook"
```

For `nnedi3` prescaler, `neurons` and `window` settings are indicated in the
filename.

For `ravu` prescaler, `radius` setting is indicated in the filename.

# About RAVU

RAVU (Rapid and Accurate Video Upscaling) is a set of prescalers inspired by
[RAISR (Rapid and Accurate Image Super Resolution)](https://ai.googleblog.com/2016/11/enhance-raisr-sharp-images-with-machine.html).
It comes with different variants to fit different scenarios.

`ravu` and `ravu-lite` upscale only luma plane (of a YUV video), which means
chroma planes will be handled by `--cscale` later. `ravu-lite` is faster and
sharper. It also introduces no half pixel offset.

`ravu-yuv` and `ravu-rgb` upscale video after all planes are merged. This happens
after `--cscale` (or other chroma prescaler) is applied. `ravu-yuv` assumes YUV
video and will fail on others (for example, PNG picture).

`ravu-zoom` is another variant which is able to upscale video to arbitrary ratio
directly. Its sharpness is close to `ravu-lite`. But it renders at target
resolution, so expect it to be much slower than `ravu` for perfect 2x upscaling.

`ravu-lite-ar` and `ravu-zoom-ar` uses [anti-ringing filter (of EWA scalers)](https://github.com/haasn/libplacebo/commit/0581828343ddaafb81d296aa510d4d141e4d9b50) from libplacebo to reduce
[ringing artifacts](https://en.wikipedia.org/wiki/Ringing_artifacts). The default anti-ringing strength in master branch is set to 0.8.

# License

Shaders in this repo are licensed under terms of LGPLv3. Check the header of
each file for details.
