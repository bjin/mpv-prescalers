This repo contains the scripts that generate user shader for prescaling in
[mpv](https://mpv.io/), currently only `nnedi3` and `superxbr` are supported.

For the generated user shaders, check the [master branch](https://github.com/bjin/mpv-prescalers/tree/master).

# Usage

Python 3 is required. Running `gen.sh` will generate all user shaders in
current directory.

Alternatively, you could generate shaders with customized options:
```
./nnedi3.py --target yuv --nns 32 --win 8x4 --max-downscaling-ratio 1.8 > ~/.config/mpv/shaders/nnedi3.hook
```

Or play video directly with scripts
```
mpv --opengl-shaders=<(path/to/superxbr.py --target=native) video.mkv
```

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
