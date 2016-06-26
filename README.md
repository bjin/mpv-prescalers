This repo contains user shaders for prescaling in [mpv](https://mpv.io/),
currently only `nnedi3` and `superxbr` are supported.

For the scripts generating these user shaders, check the [source
branch](https://github.com/bjin/mpv-prescalers/tree/source).

# Filenames

Suffix in the filename indicates the type of planes shader is hooked on:

* Without any suffix: Triggered on luma plane only (like `prescale-luma=...` option in `mpv`).
* `-chroma`: Triggered on chroma plane only.
* `-yuv`: Triggered on both luma and chroma planes.
* `-all`: Triggered on all source planes (including non-YUV formats).

For `nnedi3` prescaler, `neurons` and `window` settings are indicated in the
filename.

In addition, `superxbr-native.hook` is a native implementation of `superxbr`
which do the upscaling on RGB, and is most likely the one you want use.
`superxbr-native-yuv.hook` is a similar shader but requires the original source
to be YUV, and thus should be considered as experimental.

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

Add `user-shaders="prescaler.hook"` sub-option to `vo` settings:

```
vo=opengl-hq:...:user-shaders="~~/shaders/nnedi3-nns32-win8x4.hook"
```

All shaders are for one pass only. If you want to have `4x` upscaling, trigger
the same shader twice. For `4x` luma prescaling:

```
vo=opengl-hq:...:user-shaders="~~/shaders/nnedi3-nns32-win8x4.hook,~~/shaders/nnedi3-nns32-win8x4.hook"
```

Pay attention that for `4:2:0` sub-sampled `YUV` video, you need an additional
chroma-only prescaling pass to match the post-prescaling size of luma and
chroma planes (they are still not aligned though):

```
vo=opengl-hq:...:user-shaders="~~/shaders/nnedi3-nns32-win8x4-chroma.hook"
vo=opengl-hq:...:user-shaders="~~/shaders/nnedi3-nns32-win8x4-yuv.hook,~~/shaders/nnedi3-nns32-win8x4-chroma.hook"
```

# Known Issue

* `nnedi3-nns32-win8x6-{chroma,yuv,all}.hook` are extremely slow with nvidia
  driver, avoid using these shaders for now.

# License

Both shaders were ported from [MPDN
project](https://github.com/zachsaw/MPDN_Extensions), and were originally
licensed under terms of [LGPLv3](https://www.gnu.org/licenses/lgpl-3.0.en.html).
`superxbr` shader in addition was licensed under terms of a more permissive
license ([MIT](https://opensource.org/licenses/MIT)).

The ported shaders (in mpv) also include contributions licensed under terms of
LGPLv2 or later (particularly, major part of `superxbr` was rewritten by
@haasn).

As a whole, the shaders in this repo are licensed with LGPLv3.
