This repo contains user shaders for prescaling in [mpv](https://mpv.io/).

For the scripts generating these user shaders, check the [source
branch](https://github.com/bjin/mpv-prescalers/tree/source).

Shaders in [`gather/` directory](https://github.com/bjin/mpv-prescalers/tree/master/gather)
and [`compute/` directory](https://github.com/bjin/mpv-prescalers/tree/master/compute)
are **generally faster** but requires recent version of OpenGL.
Use these shaders only if they actually work (i.e. no blue screen and no noticeable distortion).

Shaders in [`vulkan/` directory](https://github.com/bjin/mpv-prescalers/tree/master/vulkan)
are using `rgba16hf` LUT, and required by `gpu-api=vulkan` and
`gpu-api=d3d11`. Use these shaders if you encountered the following error:

```
[vo/gpu] Unrecognized/unavailable FORMAT name: 'rgba16f'!
```

# Usage

You only need to download shaders you actually use. The following part of this
section assumes that they are in `shaders` directory in the `mpv` configure
folder (usually `~/.config/mpv/shaders` on Linux).

Use `glsl-shaders="prescaler.hook"` option to load those shaders. (This will
override other user shaders, use `glsl-shaders-append` in that case)

```
glsl-shaders="~~/shaders/ravu-r3.hook"
```

All shaders are for one pass only. If you want to have `4x` upscaling, trigger
the same shader twice. All the shaders here are generated with
`max-downscaling-ratio` set to `1.414213`. They will be disabled if upscaling is not necessary.

```
glsl-shaders-append="~~/shaders/ravu-r3.hook"
glsl-shaders-append="~~/shaders/ravu-r3.hook"
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

`ravu-3x` is just like its `ravu`/`ravu-yuv`/`ravu-rgb` counterpart. But
instead of double the size of video, it triple the size. It also requires
compute shader OpenGL capability, which means decent GPU and driver (and no
macOS support).

`ravu-chroma` is a chroma prescaler (could be considered as replacement of `--cscale`).
It uses downscaled luma plane to calculate gradient and guide chroma planes upscaling.

Due to current limitation of `mpv`'s hook system, there are some caveats for using `ravu-chroma`:

1. It won't detect chroma offset introduced by itself. So it's best practice to
   use this shader at most once.
2. You need to explicitly state the chroma location, by choosing one of those
   `chroma-left` and `chroma-center` shaders. If you don't know how to/don't
   bother to check chroma location of video, or don't watch ancient videos,
   just choose `chroma-left`. If you are using [auto-profiles.lua](https://github.com/wiiaboo/mpv-scripts/blob/master/auto-profiles.lua),
   you can use `cond:get('video-params/chroma-location','unknown')=='mpeg2/4/h264'`
   for `chroma-left` shader and `cond:get('video-params/chroma-location','unknown')=='mpeg1/jpeg'`
   for `chroma-center` shader.
3. `cscale` will still be used to correct minor offset.

`ravu-zoom` is another variant which is able to upscale video to arbitrary ratio
directly. Its sharpness is close to `ravu-lite`. But it renders at target
resolution, so expect it to be much slower than `ravu` for perfect 2x upscaling.

# Known Issue

1. `ravu-lite` is incompatible with `--fbo-format=rgb10_a2` (default
   for some OpenGL ES implementation). Use `rgba16f` or `rgba16` if available.
2. `ravu-[zoom-]r4-{rgb,yuv}` causes distortion with lower-end intel card.

# License

Shaders in this repo are licensed under terms of LGPLv3. Check the header of
each file for details.
