#!/usr/bin/env python3
#
# Copyright (C) 2019 Bin Jin <bjin@ctrl-d.org>
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

import numpy as np

import userhook

class FP16_CNN(userhook.UserHook):

    def __init__(self, weights_file, **args):
        super().__init__(**args)

        # TODO read weights
        self.weights = []
        self.weights_ptr = 0

    def __del__(self):
        assert self.weights_ptr == len(self.weights), "not all weights are consumed"

    def next_weight(self):
        assert self.weights_ptr < len(self.weights), "weights are all consumed"

        ret = self.weights[self.weights_ptr]
        self.weights_ptr += 1

        return "%shf" % np.float16(ret)

    def next_weights(self, num):
        ret = []
        for i in range(num):
            ret.append(self.next_weight())
        return ret

    def setup(self):
        self.set_skippable(2.0, 2.0)
        self.add_mappings(
            float="float16_t",
            vec4="f16vec4",
            mat4="f16mat4",
            zero="0.0hf")

    def encode(self, hf4a, hf4b):
        encode1 = lambda x: "uintBitsToFloat(packFloat2x16(%s))" % x
        encodes = encode1(hf4a + ".xy"), encode1(hf4a + ".zw"), encode1(hf4b + ".xy"), encode1(hf4b + ".zw")
        return "vec4(%s, %s, %s, %s)" % encodes

    def decode(self, f4, comp):
        decode1 = lambda x: "unpackFloat2x16(floatBitsToUint(%s))" % x
        comps = ["xy", "zw"][comp]
        decodes = decode1(f4 + "." + comps[0]), decode1(f4 + "." + comps[1])
        return "$vec4(%s, %s)" % decodes

    def generate_feature_layer(self, dest):
        self.reset()
        GLSL = self.add_glsl

        bias = []
        trans = [[] for i in range(25)]

        for i in range(2):
            bias.extend(self.next_weights(4))
            for j in range(25):
                trans[j].extend(self.next_weights(4))

        self.set_description("feature")
        self.setup()
        self.save_tex(dest)
        self.set_components(4)

        GLSL("vec4 hook() {")

        GLSL("$vec4 res0=$vec4(%s), res1=$vec4(%s);" %
                (",".join(bias[:4]), ",".join(bias[4:])))
        GLSL("$float tmp;")

        for i in range(5):
            for j in range(5):
                trans_ij = trans[i * 5 + j]
                GLSL("tmp=$float(HOOKED_texOff(vec2(%d.0, %d.0)));" % (i - 2, j - 2))
                GLSL("res0+=$vec4(%s)*tmp;" % ",".join(trans_ij[:4]))
                GLSL("res1+=$vec4(%s)*tmp;" % ",".join(trans_ij[4:]))

        GLSL("return %s;" % self.encode("res0", "res1"))

        GLSL("}")

        return super().generate()

    def generate_mapping_layer(self, src, dest):
        self.reset()
        GLSL = self.add_glsl

        bias = []
        trans = [[[] for j in range(2)] for i in range(9)]
        thr = []

        for i in range(2):
            bias.extend(self.next_weights(4))
            for j in range(9):
                for k in range(2):
                    trans[j][k].extend(self.next_weights(16))
            thr.extend(self.next_weights(4))

        self.set_description("mapping")
        self.setup()
        self.bind_tex(src)
        self.save_tex(dest)
        self.set_components(4)

        GLSL("vec4 hook() {")

        GLSL("$vec4 res0=$vec4(%s), res1=$vec4(%s);" %
                (",".join(bias[:4]), ",".join(bias[4:])))
        GLSL("vec4 tmp;")
        GLSL("$vec4 tmp2;")

        for i in range(3):
            for j in range(3):
                trans_ij = trans[i * 3 + j]
                GLSL("tmp=%s_texOff(vec2(%d.0, %d.0));" % (src, i - 1, j - 1))
                for k in range(2):
                    GLSL("tmp2=%s;" % self.decode("tmp", k))
                    GLSL("res0+=$mat4(%s)*tmp2;" % ",".join(trans_ij[k][:16]))
                    GLSL("res1+=$mat4(%s)*tmp2;" % ",".join(trans_ij[k][16:]))

        GLSL("res0=max(res0,$vec4($zero))+$vec4(%s)*min(res0,$vec4($zero));" % ",".join(thr[:4]))
        GLSL("res1=max(res1,$vec4($zero))+$vec4(%s)*min(res1,$vec4($zero));" % ",".join(thr[4:]))

        GLSL("return %s;" % self.encode("res0", "res1"))

        GLSL("}")

        return super().generate()

    def generate_residual_layer(self, src, src2, dest):
        self.reset()
        GLSL = self.add_glsl

        bias = []
        trans = [[] for j in range(2)]
        thr = []

        for i in range(2):
            bias.extend(self.next_weights(4))
            for j in range(2):
                trans[j].extend(self.next_weights(16))
            thr.extend(self.next_weights(4))

        self.set_description("residual")
        self.setup()
        self.bind_tex(src)
        self.bind_tex(src2)
        self.save_tex(dest)
        self.set_components(4)

        GLSL("vec4 hook() {")
        GLSL("$vec4 res0=$vec4(%s), res1=$vec4(%s);" %
                (",".join(bias[:4]), ",".join(bias[4:])))
        GLSL("vec4 tmp;")
        GLSL("$vec4 tmp2;")
        GLSL("tmp=%s_texOff(vec2(0.0, 0.0));" % src)
        for k in range(2):
            GLSL("tmp2=%s;" % self.decode("tmp", k))
            GLSL("res0+=$mat4(%s)*tmp2;" % ",".join(trans[k][:16]))
            GLSL("res1+=$mat4(%s)*tmp2;" % ",".join(trans[k][16:]))
        GLSL("tmp=%s_texOff(vec2(0.0, 0.0));" % src2)
        GLSL("res0+=%s;" % self.decode("tmp", 0))
        GLSL("res1+=%s;" % self.decode("tmp", 1))

        GLSL("res0=max(res0,$vec4($zero))+$vec4(%s)*min(res0,$vec4($zero));" % ",".join(thr[:4]))
        GLSL("res1=max(res1,$vec4($zero))+$vec4(%s)*min(res1,$vec4($zero));" % ",".join(thr[4:]))

        GLSL("return %s;" % self.encode("res0", "res1"))

        GLSL("}")

        return super().generate()

    def generate_subconv_layer(self, src, dest):
        self.reset()
        GLSL = self.add_glsl

        bias = []
        trans = [[] for j in range(9)]

        bias = self.next_weights(4)
        for i in range(9):
            for k in range(2):
                trans[i].append(self.next_weights(16))

        self.set_description("subconv")
        self.setup()
        self.bind_tex(src)
        self.save_tex(dest)
        self.set_components(4)

        GLSL("vec4 hook() {")
        GLSL("$vec4 res=$vec4(%s);" % ",".join(bias))

        GLSL("vec4 tmp;")
        GLSL("$vec4 tmp2;")

        for i in range(3):
            for j in range(3):
                trans_ij = trans[i * 3 + j]
                GLSL("tmp=%s_texOff(vec2(%d.0, %d.0));" % (src, i - 1, j - 1))
                for k in range(2):
                    GLSL("tmp2=%s;" % self.decode("tmp", k))
                    GLSL("res+=$mat4(%s)*tmp2;" % ",".join(trans_ij[k]))

        GLSL("return vec4(res);")

        GLSL("}")

        return super().generate()

    def generate_aggregation_layer(self, src):
        self.reset()
        GLSL = self.add_glsl

        self.set_description("aggregation")
        self.setup()
        self.bind_tex(src)
        self.set_components(1)

        GLSL("""
vec4 hook() {
    vec2 dir = fract(HOOKED_pos * HOOKED_size) - 0.5;
    int idx = int(dir.x > 0.0) * 2 + int(dir.y > 0.0);
    return vec4(%s_texOff(-dir)[idx], 0.0, 0.0, 0.0);
}
""" % src)

        return super().generate()

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="generate fp16-CNN user shader for mpv")
    parser.add_argument(
        '-w',
        '--weights-file',
        nargs=1,
        required=True,
        type=str,
        help='weights file name')
    parser.add_argument(
        '-r',
        '--max-downscaling-ratio',
        nargs=1,
        type=float,
        default=[1.414],
        help='allowed downscaling ratio (default: 1.414)')

    args = parser.parse_args()
    weights_file = args.weights_file[0]
    max_downscaling_ratio = args.max_downscaling_ratio[0]

    gen = FP16_CNN(hook=["LUMA"],
                   weights_file=weights_file,
                   target_tex="OUTPUT",
                   max_downscaling_ratio=max_downscaling_ratio)

    feature = "FEATURE"
    model1 = "MODEL1"
    model2 = "MODEL2"
    res = "RES"
    subconv = "SUBCONV"
    sys.stdout.write(gen.generate_feature_layer(dest=feature))
    sys.stdout.write(gen.generate_mapping_layer(src=feature, dest=model1))
    sys.stdout.write(gen.generate_mapping_layer(src=model1, dest=model2))
    sys.stdout.write(gen.generate_mapping_layer(src=model2, dest=model1))
    sys.stdout.write(gen.generate_mapping_layer(src=model1, dest=model2))
    sys.stdout.write(gen.generate_residual_layer(src=model2, src2=feature, dest=res))
    sys.stdout.write(gen.generate_subconv_layer(src=res, dest=subconv))
    sys.stdout.write(gen.generate_aggregation_layer(src=subconv))
