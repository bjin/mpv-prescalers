#!/bin/sh

set -e

DIR="$(dirname "$0")"

max_downscaling_ratio=1.6

for target in luma chroma yuv; do
    suffix="-$target"
    [ "$target" = "luma" ] && suffix=""
    for nns in 16 32 64 128 256; do
        for win in 8x4 8x6; do
            file_name="nnedi3-nns$nns-win$win$suffix.hook"
            "$DIR/nnedi3.py" --target "$target" --nns "$nns" --win "$win" --max-downscaling-ratio "$max_downscaling_ratio" > "$file_name"
            if [ -d gather ]; then
                "$DIR/nnedi3.py" --target "$target" --nns "$nns" --win "$win" --max-downscaling-ratio "$max_downscaling_ratio" --use-gather > "gather/$file_name"
            fi
        done
    done
done

for target in luma yuv rgb; do
    suffix="-$target"
    [ "$target" = "luma" ] && suffix=""
    file_name="superxbr$suffix.hook"
    "$DIR/superxbr.py" --target "$target" > "$file_name"
done

for target in luma chroma-left chroma-center yuv rgb; do
    suffix="-$target"
    [ "$target" = "luma" ] && suffix=""
    for radius in 2 3 4; do
        file_name="ravu-r$radius$suffix.hook"
        weights_file="$DIR/ravu_weights-r$radius.py"
        "$DIR/ravu.py" --target "$target" --weights-file "$weights_file" --max-downscaling-ratio "$max_downscaling_ratio" > "$file_name"
        if [ -d gather -a \( "$target" = "luma" -o "$target" = "chroma-left" -o "$target" = "chroma-center" \) ]; then
            "$DIR/ravu.py" --target "$target" --weights-file "$weights_file" --max-downscaling-ratio "$max_downscaling_ratio" --use-gather > "gather/$file_name"
        fi
        if [ -d compute ]; then
            "$DIR/ravu.py" --target "$target" --weights-file "$weights_file" --max-downscaling-ratio "$max_downscaling_ratio" --use-compute-shader > "compute/$file_name"
        fi
    done
done

for radius in 2 3 4; do
    file_name="ravu-lite-r$radius.hook"
    weights_file="$DIR/ravu-lite_weights-r$radius.py"
    "$DIR/ravu-lite.py" --weights-file "$weights_file" --max-downscaling-ratio "$max_downscaling_ratio" > "$file_name"
    if [ -d gather ]; then
        "$DIR/ravu-lite.py" --weights-file "$weights_file" --max-downscaling-ratio "$max_downscaling_ratio" --use-gather > "gather/$file_name"
    fi
    if [ -d compute ]; then
        "$DIR/ravu-lite.py" --weights-file "$weights_file" --max-downscaling-ratio "$max_downscaling_ratio" --use-compute-shader > "compute/$file_name"
    fi
done
