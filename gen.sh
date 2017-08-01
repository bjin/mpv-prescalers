#!/bin/sh

set -e

DIR="$(dirname "$0")"

max_downscaling_ratio=1.414213

gen_ravu() {
    float_format="$1"
    for target in luma chroma-left chroma-center yuv rgb; do
        suffix="-$target"
        [ "$target" = "luma" ] && suffix=""
        for radius in 2 3 4; do
            file_name="ravu-r$radius-ar1$suffix.hook"
            weights_file="$DIR/weights/ravu_weights-r$radius.py"
            "$DIR/ravu.py" --target "$target" --weights-file "$weights_file" --max-downscaling-ratio "$max_downscaling_ratio" --float-format "$float_format" --smooth > "$file_name"
            if [ -d gather ]; then
                "$DIR/ravu.py" --target "$target" --weights-file "$weights_file" --max-downscaling-ratio "$max_downscaling_ratio" --float-format "$float_format" --smooth --use-gather > "gather/$file_name"
            fi
            if [ -d compute ]; then
                "$DIR/ravu.py" --target "$target" --weights-file "$weights_file" --max-downscaling-ratio "$max_downscaling_ratio" --float-format "$float_format" --smooth --use-compute-shader > "compute/$file_name"
            fi
        done
    done
}

gen_ravu float16gl
if [ -d vulkan ]; then
    cd vulkan
    gen_ravu float16vk
fi
