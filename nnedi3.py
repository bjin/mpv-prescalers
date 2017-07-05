#!/usr/bin/env python3

#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import enum
import os
import struct

import userhook

#
# Copyright (c) 2016 mpv developers <mpv-team@googlegroups.com>
#

#
# NNEDI3, an intra-field deinterlacer
#
# The original filter was authored by Kevin Stone (aka. tritical) and is
# licensed under GPL2 terms:
#     http://bengal.missouri.edu/~kes25c/
#
# A LGPLv3 licensed OpenCL kernel was created by SEt:
#     http://forum.doom9.org/showthread.php?t=169766
#
# A HLSL port further modified by madshi, Shiandow and Zach Saw could be
# found at (also LGPLv3 licensed):
#     https://github.com/zachsaw/MPDN_Extensions
#


class Neurons(enum.Enum):
    nns16 = 0
    nns32 = 1
    nns64 = 2
    nns128 = 3
    nns256 = 4

    def get_neurons(self):
        return [16, 32, 64, 128, 256][self.value]


class Window(enum.Enum):
    win8x4 = 0
    win8x6 = 1

    def get_width(self):
        return 8

    def get_height(self):
        return [4, 6][self.value]


class Step(enum.Enum):
    double_y = 0
    double_x = 1


class NNEDI3(userhook.UserHook):

    weight_offsets = [0, 1088, 3264, 7616, 16320, 33728, 35328, 38528, 44928, 57728]
    weights = None

    weights_file = "nnedi3_weights.bin"
    weights_filesize = 83328 * 4
    weights_dirs = [os.path.dirname(os.path.realpath(__file__)),
                    os.path.realpath(os.getcwd())]

    weight_fmt = struct.Struct("<i")
    assert weight_fmt.size == 4

    def __init__(self, neurons, window, **args):
        super().__init__(**args)

        self.neurons = neurons.get_neurons()
        self.window_width = window.get_width()
        self.window_height = window.get_height()
        self.offset = NNEDI3.weight_offsets[window.value * len(Neurons) +
                                            neurons.value]

    @staticmethod
    def load_weights():
        if NNEDI3.weights:
            return
        for weights_dir in NNEDI3.weights_dirs:
            try:
                NNEDI3.weights = open(
                    os.path.join(weights_dir, NNEDI3.weights_file),
                    "rb").read()
                assert len(NNEDI3.weights) == NNEDI3.weights_filesize
                return
            except IOError:
                pass
        raise Exception("unable to load %s" % NNEDI3.weights_file)

    @staticmethod
    def weight_at(ptr):
        return NNEDI3.weight_fmt.unpack_from(NNEDI3.weights, ptr * 4)[0]

    def generate(self, step):
        self.load_weights()
        self.reset()
        GLSL = self.add_glsl

        width = self.window_width
        height = self.window_height

        self.set_description("NNEDI3 (step=%s, window=%dx%d, neurons=%d)" %
                             (step.name, width, height, self.neurons))

        assert width % 4 == 0
        sample_count = width * height // 4

        GLSL('#pragma optionNV(fastprecision on))')

        GLSL("""
float nnedi3(int comp) {""")

        if step == Step.double_y:
            self.set_transform(1, 2, 0, -0.5, True)
            GLSL("""
if ((transpose(HOOKED_rot) * fract(HOOKED_pos * HOOKED_size)).y < 0.5)
    return HOOKED_texOff(vec2(0, 0.25))[comp];
#define GET(i, j) HOOKED_texOff(vec2((i)-(%f),(j)-(%f)+0.25))[comp]""" %
                 (width / 2.0 - 1, (height - 1) / 2.0))
        elif step == Step.double_x:
            self.set_transform(2, 1, -0.5, 0, True)
            GLSL("""
if (fract(HOOKED_pos.x * HOOKED_size.x) < 0.5)
    return HOOKED_texOff(vec2(0.25, 0))[comp];
#define GET(i, j) HOOKED_texOff(vec2((j)-(%f)+0.25,(i)-(%f)))[comp]""" % (
                (height - 1) / 2.0, width / 2.0 - 1))
        else:
            raise Exception("unknown step: %s" % repr(step))

        GLSL("""
vec4 samples[%d];""" % sample_count)

        for y in range(0, height):
            for x in range(0, width, 4):
                GLSL("""
samples[%d] = vec4(GET(%d.0, %d.0), GET(%d.0, %d.0), \
GET(%d.0, %d.0), GET(%d.0, %d.0));""" % ((y * width + x) / 4, x, y, x + 1, y,
                                         x + 2, y, x + 3, y))
        GLSL("""
float sum = 0.0, sumsq = 0.0;
for (int i = 0; i < %d; i++) {
    sum += dot(samples[i], vec4(1.0));
    sumsq += dot(samples[i], samples[i]);
}""" % sample_count)

        GLSL("""
float mstd0 = sum / %d.0;
float mstd1 = sumsq / %d.0 - mstd0 * mstd0;
float mstd2 = mix(0.0, inversesqrt(mstd1), mstd1 >= %s);
mstd1 *= mstd2;""" % (width * height, width * height, "1.192092896e-7"))

        GLSL("""
float vsum = 0.0, wsum = 0.0, sum1, sum2;
#define T(x) intBitsToFloat(x)
#define W(i,w0,w1,w2,w3) dot(samples[i],vec4(T(w0),T(w1),T(w2),T(w3)))
#define WS(w0,w1) \
sum1 = exp(sum1 * mstd2 + T(w0)); \
sum2 = sum2 * mstd2 + T(w1); \
wsum += sum1; \
vsum += sum1*(sum2/(1.0+abs(sum2)));""")

        for n in range(self.neurons):
            ptr = self.offset + (sample_count * 2 + 1) * 4 * n
            line = []
            for s in range(2):
                line.append("sum%d" % (s + 1))
                for i in range(sample_count):
                    line.append("%sW(%d,%d,%d,%d,%d)" % (
                        "=" if i == 0 else "+",
                        i,
                        self.weight_at(ptr),
                        self.weight_at(ptr + 1),
                        self.weight_at(ptr + 2),
                        self.weight_at(ptr + 3))) # yapf: disable
                    ptr += 4
                line.append(";")
            line.append("WS(%d,%d);" %
                        (self.weight_at(ptr), self.weight_at(ptr + 1)))
            GLSL("".join(line))

        GLSL("""
return clamp(mstd0 + 5.0 * vsum / wsum * mstd1, 0.0, 1.0);
}  // nnedi3""")

        comps = self.max_components()
        GLSL("""
vec4 hook() {
    return vec4(%s);
}""" % ", ".join("nnedi3(%d)" % i if i < comps else "0.0" for i in range(4)))

        return super().generate()


if __name__ == "__main__":
    import argparse
    import sys

    hooks = {"luma": ["LUMA"],
             "chroma": ["CHROMA"],
             "yuv": ["LUMA", "CHROMA"],
             "all": ["LUMA", "CHROMA", "RGB", "XYZ"]}

    neurons = {16: Neurons.nns16,
               32: Neurons.nns32,
               64: Neurons.nns64,
               128: Neurons.nns128,
               256: Neurons.nns256}

    windows = {"8x4": Window.win8x4, "8x6": Window.win8x6}

    parser = argparse.ArgumentParser(
        description="generate NNEDI3 user shader for mpv")
    parser.add_argument('-t',
                        '--target',
                        nargs=1,
                        choices=sorted(hooks.keys()),
                        default=["luma"],
                        help='target that shader is hooked on (default: luma)')
    parser.add_argument('-n',
                        '--nns',
                        nargs=1,
                        type=int,
                        choices=sorted(neurons.keys()),
                        default=[32],
                        help='neurons for NNEDI3 (default: 32)')
    parser.add_argument('-w',
                        '--win',
                        nargs=1,
                        choices=sorted(windows.keys()),
                        default=["8x4"],
                        help='sampling window size of NNEDI3 (default: 8x4)')
    parser.add_argument('-r',
                        '--max-downscaling-ratio',
                        nargs=1,
                        type=float,
                        default=[None],
                        help='allowed downscaling ratio (default: no limit)')

    args = parser.parse_args()
    hook = hooks[args.target[0]]
    neuron = neurons[args.nns[0]]
    window = windows[args.win[0]]
    max_downscaling_ratio = args.max_downscaling_ratio[0]

    target_tex = "LUMA" if hooks == ["CHROMA"] else "OUTPUT"
    gen = NNEDI3(neuron,
                 window,
                 hook=hook,
                 target_tex=target_tex,
                 max_downscaling_ratio=max_downscaling_ratio)

    sys.stdout.write(userhook.LICENSE_HEADER)
    for step in list(Step):
        sys.stdout.write(gen.generate(step))
