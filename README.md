This gist contains user shaders for prescaling in [mpv](https://mpv.io/),
currently only `nnedi3` and `superxbr` is supported.

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

For example:
* `nnedi3-nns32-win8x4.hook` is the user shader for luma `nnedi3` prescaling
  with `32` neurons and a local sampling window size of `8x4`.
* `superxbr-chroma.hook` is the user shader for chroma-only `superxbr`
  prescaling.
* `nnedi3-nns128-win8x6-all.hook` is the user shader for `nnedi3` prescaling
  on all planes with `128` neurons and a local sampling window size of `8x6`.

# Usage

You only need to download shaders you actually use. The following part of this
section assumes that they are in `shaders` directory in the `mpv` configure
folder (usually `~/.config/mpv/shaders` on Linux).

Unlike the `prescale-luma` option in `mpv`, these shaders will be triggered
**regardless of the video size and screen resolution**. They are better guarded by
profiles meant for different type of video (or with help from some lua script).

Add `user-shaders="prescaler.hook"` suboption to `vo` settings:

```
vo=opengl-hq:...:user-shaders="~~/shaders/nnedi3-nns32-win8x4.hook"
```

All shaders are for one pass only. If you want to have `4x` upscaling, trigger
the same shader twice. For `4x` luma prescaling:

```
vo=opengl-hq:...:user-shaders="~~/shaders/nnedi3-nns32-win8x4.hook,nnedi3-nns32-win8x4.hook"
```

Pay attention that for `4:2:0` sub-sampled `YUV` video, you need an additional
chroma-only prescaling pass to match the post-prescaling size of luma and
chroma planes (they are still not aligned though):

```
vo=opengl-hq:...:user-shaders="~~/shaders/nnedi3-nns32-win8x4-chroma.hook"
vo=opengl-hq:...:user-shaders="~~/shaders/nnedi3-nns32-win8x4-yuv.hook,nnedi3-nns32-win8x4-chroma.hook"
```

# Known Issue

* `nnedi3-nns32-win8x6-{chroma,yuv,all}.hook` are extremely slow with nvidia
  driver, avoid using these shaders for now.

* `prescale-downscaling-threshold=1.6` is used, and cannot be customized other
  than modifying the shader manually.

# License

Both shaders are ported from [MPDN
project](https://github.com/zachsaw/MPDN_Extensions), and are originally
licensed under terms of [LGPLv3](https://www.gnu.org/licenses/lgpl-3.0.en.html).
`superxbr` shader in additon is licensed under terms of a more permissive
license ([MIT](https://opensource.org/licenses/MIT)).

The ported shaders (in mpv) also include contributions licensed under terms of
LGPLv2 or later (particularly, major part of `superxbr` was refactored by
@haasn).

As a whole, the shaders in this gist are licensed under terms of LGPLv3.
