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
    step1 = 0
    step2 = 1
    step3 = 2
    step4 = 3


class Profile(enum.Enum):
    luma = 0
    chroma = 1
    yuv = 2


class NNEDI3(userhook.UserHook):

    weight_offsets = [0, 1088, 3264, 7616, 16320, 33728, 35328, 38528, 44928, 57728]
    weights = None

    weights_file = "nnedi3_weights.bin"
    weights_filesize = 83328 * 4
    weights_dirs = [os.path.dirname(os.path.realpath(__file__)),
                    os.path.realpath(os.getcwd())]

    weight_fmt = struct.Struct("<i")
    assert weight_fmt.size == 4

    def __init__(self, profile, neurons, window, int_tex_name = "nnedi3_int", **args):
        super().__init__(**args)

        self.profile = profile
        self.neurons = neurons.get_neurons()
        self.window_width = window.get_width()
        self.window_height = window.get_height()
        self.offset = NNEDI3.weight_offsets[window.value * len(Neurons) +
                                            neurons.value]
        self.int_tex_name = int_tex_name

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

    def weightW(self, n, s, x, y):
        window_size = self.window_width * self.window_height
        ptr = self.offset + \
              (window_size * 2 + 4) * n + window_size * s + \
              x + y * self.window_width
        return self.weight_at(ptr)

    def weightWS(self, n, s, i):
        window_size = self.window_width * self.window_height
        ptr = self.offset + \
              (window_size * 2 + 4) * n + window_size * 2 + \
              i
        return self.weight_at(ptr)

    def generate(self, step, use_gather=False):
        self.load_weights()
        self.reset()
        GLSL = self.add_glsl

        width = self.window_width
        height = self.window_height

        self.set_description("NNEDI3 (%s, %s, nns%d, win%dx%d)" %
                             (step.name, self.profile.name, self.neurons, width, height))

        assert width % 2 == 0 and height % 2 == 0
        sample_count = width * height // 4

        GLSL('#pragma optionNV(fastprecision on)')

        tex_name = [["HOOKED", self.int_tex_name + "01"],
                    [self.int_tex_name + "10", self.int_tex_name + "11"]]

        # This checks against all passes, and works since "HOOKED" is same for
        # all of them.
        self.set_skippable(2, 2)

        if step == Step.step4:
            self.set_transform(2, 2, -0.5, -0.5)

            self.bind_tex(tex_name[0][1])
            self.bind_tex(tex_name[1][0])
            self.bind_tex(tex_name[1][1])

            # FIXME: get rid of branching (is it even possible?)
            GLSL("""
vec4 hook() {
    vec2 dir = fract(HOOKED_pos * HOOKED_size) - 0.5;
    if (dir.x < 0) {
        if (dir.y < 0)
            return %s_texOff(-dir);
        return %s_texOff(-dir);
    } else {
        if (dir.y < 0)
            return %s_texOff(-dir);
        return %s_texOff(-dir);
    }
}
""" % (tex_name[0][0], tex_name[0][1], tex_name[1][0], tex_name[1][1]))

            return super().generate()

        if self.profile == Profile.luma:
            components = 1
        elif self.profile == Profile.chroma:
            components = 2
        elif self.profile == Profile.yuv:
            components = 3
            self.assert_native_yuv()

        center_x = (width // 2 - 1) * 2
        center_y = (height // 2 - 1) * 2 + 1

        if step == Step.step1:
            offset_x, offset_y = 0, 1
            get_position = lambda x, y: (x * 2 - center_x, y * 2 - center_y)
        else:
            self.bind_tex(tex_name[0][1])
            if step == Step.step2:
                offset_x, offset_y = 1, 0
            elif step == Step.step3:
                offset_x, offset_y = 1, 1
            get_position = lambda x, y: (y * 2 - center_y, x - center_x // 2)

        self.save_tex(tex_name[offset_x][offset_y])

        sample_positions = {}
        for y in range(height):
            for x in range(width):
                nx, ny = get_position(x, y)
                nx += offset_x
                ny += offset_y
                tex = tex_name[nx % 2][ny % 2]
                sample_positions.setdefault(tex, {})[nx // 2, ny // 2] = x, y

        gather_offsets = [(0, 1), (1, 1), (1, 0), (0, 0)]

        sampling_info = []
        for tex in sorted(sample_positions.keys()):
            mapping = sample_positions[tex]
            while len(mapping) > 0:
                base = min(mapping.keys())
                global_pos = []
                window_pos = []
                for dx, dy in gather_offsets:
                    npos = base[0] + dx, base[1] + dy
                    global_pos.append(npos)
                    window_pos.append(mapping.pop(npos))
                sampling_info.append((tex, global_pos, window_pos))

        assert len(sampling_info) == sample_count

        GLSL("""
float nnedi3(vec4 samples[%d]) {""" % sample_count)


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
            line = []
            for s in range(2):
                line.append("sum%d" % (s + 1))
                for i in range(sample_count):
                    tex, global_pos, window_pos = sampling_info[i]
                    weights = []
                    for x, y in window_pos:
                        weights.append(self.weightW(n, s, x, y))
                    line.append("%sW(%d,%d,%d,%d,%d)" % (
                        "=" if i == 0 else "+",
                        i, weights[0], weights[1], weights[2], weights[3]))
                line.append(";")
            line.append("WS(%d,%d);" %
                        (self.weightWS(n, s, 0), self.weightWS(n, s, 1)))
            GLSL("".join(line))

        GLSL("""
return clamp(mstd0 + 5.0 * vsum / wsum * mstd1, 0.0, 1.0);
}  // nnedi3""")

        GLSL("""
vec4 hook() {""")

        GLSL("vec4 ret = vec4(0.0);")
        for comp in range(components):
            GLSL("vec4 samples%d[%d];" % (comp, sample_count))
            for i in range(sample_count):
                tex, global_pos, window_pos = sampling_info[i]
                if use_gather:
                    base = min(global_pos)
                    to_fetch = "%s_mul * textureGatherOffset(%s_raw, %s_pos, ivec2(%d, %d), %d)"
                    to_fetch = to_fetch % (tex, tex, tex, base[0], base[1], comp)
                    GLSL("samples%d[%d] = %s;" % (comp, i, to_fetch))
                else:
                    for j, pos in enumerate(global_pos):
                        to_fetch = "%s_texOff(vec2(%d.0, %d.0))[%d]"
                        to_fetch = to_fetch % (tex, pos[0], pos[1], comp)
                        GLSL("samples%d[%d][%d] = %s;" % (comp, i, j, to_fetch))
            GLSL("ret[%d] = nnedi3(samples%d);" % (comp, comp))

        GLSL("""
    return ret;
}  // hook""")

        return super().generate()


if __name__ == "__main__":
    import argparse
    import sys

    profile_mapping = {
        "luma": (["LUMA"], Profile.luma),
        "chroma": (["CHROMA"], Profile.chroma),
        "native-yuv": (["NATIVE"], Profile.yuv)
    }

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
                        choices=sorted(profile_mapping.keys()),
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
    parser.add_argument('--use-gather',
                        action='store_true',
                        help="enable use of textureGatherOffset (requires OpenGL 4.0)")

    args = parser.parse_args()
    hook, profile = profile_mapping[args.target[0]]
    neuron = neurons[args.nns[0]]
    window = windows[args.win[0]]
    max_downscaling_ratio = args.max_downscaling_ratio[0]
    use_gather = args.use_gather

    target_tex = "LUMA" if profile == Profile.chroma else "OUTPUT"
    gen = NNEDI3(profile,
                 neuron,
                 window,
                 hook=hook,
                 target_tex=target_tex,
                 max_downscaling_ratio=max_downscaling_ratio)

    sys.stdout.write(userhook.LICENSE_HEADER)
    for step in list(Step):
        sys.stdout.write(gen.generate(step, use_gather=use_gather))
