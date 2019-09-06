
This repo contains the training code for RAVU (and its variants).

The following guide works for `ravu`, `ravu-lite` and `ravu-3x` only.

### Settings

Change the source code directly. The following constants are configurable:

```cpp
const int radius;
const int gradient_radius;
const int quant_angle;
const int quant_strength;
const int quant_coherence;

const double min_strength[quant_strength - 1];
const double min_coherence[quant_coherence - 1];
```

### Dataset

High resolution lossless pictures are preferred to be used as training dataset.
PGM format is used for input. You can use Imagemagick convert the dataset to
ASCII/binary PGM files.

For `ravu` and `ravu-3x`, ASCII based PGM is used.

```sh
for i in *.png; do convert -depth 16 "$i" -colorspace gray -compress none "$(basename "$i" .png).pnm"; done
```

For `ravu-lite`, binary PGM is used.

```sh
for i in *.png; do convert -depth 16 "$i" -colorspace gray "$(basename "$i" .png).pnm"; done
```

### Train

```sh
./ravu train path/to/dataset/*.pnm > weights.py
```

Alternatively, you can use multiple processes to utilize all CPU cores.

```sh
find path/to/dataset -name \*.pnm -print0 | xargs -0 -n 1 -P $(nproc) ./ravu process
./ravu train path/to/dataset/*.pnm > weights.py
```

### Evaluation

For `ravu` and `ravu-lite` only.

```sh
./ravu predict weights.py path/to/dataset/*.pnm
```

### License

All code are licensed under BSD3 license.
