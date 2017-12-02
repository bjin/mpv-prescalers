#!/usr/bin/env python3
#
# Copyright (C) 2017 Bin Jin <bjin@ctrl-d.org>
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
import math

import userhook


class FloatFormat(enum.Enum):
    float16gl = 0
    float16vk = 1
    float32 = 2

class RAVU_3x(userhook.UserHook):
    """
    3x upscaling variant of RAVU-Lite, compute shader only
    """

    def __init__(self,
                 weights_file=None,
                 lut_name="ravu_3x_lut",
                 int_tex_name="ravu_3x_int",
                 **args):
        super().__init__(**args)

        exec(open(weights_file).read())

        self.radius = locals()['radius']
        self.gradient_radius = locals()['gradient_radius']
        self.quant_angle = locals()['quant_angle']
        self.quant_strength = locals()['quant_strength']
        self.quant_coherence = locals()['quant_coherence']
        self.min_strength = locals()['min_strength']
        self.min_coherence = locals()['min_coherence']
        self.gaussian = locals()['gaussian']
        self.model_weights = locals()['model_weights']

        assert len(self.min_strength) + 1 == self.quant_strength
        assert len(self.min_coherence) + 1 == self.quant_coherence

        self.lut_name = "%s%d" % (lut_name, self.radius)
        self.int_tex_name = int_tex_name

        n = self.radius * 2 - 1

        self.lut_height = self.quant_angle * self.quant_strength * self.quant_coherence
        self.lut_width = n * n + 1

    def generate_tex(self, float_format=FloatFormat.float32):
        import struct

        tex_format, item_format_str = {
            FloatFormat.float16gl: ("rgba16f", 'f'),
            FloatFormat.float16vk: ("rgba16hf", 'e'),
            FloatFormat.float32:   ("rgba32f", 'f')
        }[float_format]

        weights = []
        for i in range(self.quant_angle):
            for j in range(self.quant_strength):
                for k in range(self.quant_coherence):
                    w = self.model_weights[i][j][k]
                    for pos in range(self.lut_width // 2):
                        for z in range(8):
                            assert abs(w[z][pos] - w[~z][~pos]) < 1e-6, "filter kernel is not symmetric"
                            weights.append(w[z][pos])
        assert len(weights) == self.lut_width * self.lut_height * 4
        weights_raw = struct.pack('<%d%s' % (len(weights), item_format_str), *weights).hex()

        headers = [
            "//!TEXTURE %s" % self.lut_name,
            "//!SIZE %d %d" % (self.lut_width, self.lut_height),
            "//!FORMAT %s" % tex_format,
            "//!FILTER NEAREST"
        ]

        return "\n".join(headers + [weights_raw, ""])

    def extract_key(self, samples_list):
        GLSL = self.add_glsl
        n = self.radius * 2 - 1

        # Calculate local gradient
        gradient_left = self.radius - self.gradient_radius
        gradient_right = n - gradient_left

        GLSL("vec3 abd = vec3(0.0);")
        GLSL("float gx, gy;")
        for i in range(gradient_left, gradient_right):
            for j in range(gradient_left, gradient_right):

                def numerial_differential(f, x):
                    if x == 0:
                        return "(%s-%s)" % (f(x + 1), f(x))
                    if x == n - 1:
                        return "(%s-%s)" % (f(x), f(x - 1))
                    return "(%s-%s)/2.0" % (f(x + 1), f(x - 1))

                GLSL("gx = %s;" % numerial_differential(
                    lambda i2: samples_list[i2 * n + j], i))
                GLSL("gy = %s;" % numerial_differential(
                    lambda j2: samples_list[i * n + j2], j))
                gw = self.gaussian[i - gradient_left][j - gradient_left]
                GLSL("abd += vec3(gx * gx, gx * gy, gy * gy) * %s;" % gw)

        # Eigenanalysis of gradient matrix
        eps = "1.192092896e-7"
        GLSL("""
float a = abd.x, b = abd.y, d = abd.z;
float T = a + d, D = a * d - b * b;
float delta = sqrt(max(T * T / 4.0 - D, 0.0));
float L1 = T / 2.0 + delta, L2 = T / 2.0 - delta;
float sqrtL1 = sqrt(L1), sqrtL2 = sqrt(L2);
float theta = mix(mod(atan(L1 - a, b) + %s, %s), 0.0, abs(b) < %s);
float lambda = sqrtL1;
float mu = mix((sqrtL1 - sqrtL2) / (sqrtL1 + sqrtL2), 0.0, sqrtL1 + sqrtL2 < %s);
""" % (math.pi, math.pi, eps, eps))

        # Extract convolution kernel based on quantization of (angle, strength, coherence)
        def quantize(var_name, seps, l, r):
            if l == r:
                return "%d.0" % l
            m = (l + r) // 2
            return "mix(%s, %s, %s >= %s)" % (quantize(var_name, seps, l, m),
                                              quantize(var_name, seps, m + 1, r),
                                              var_name,
                                              seps[m])

        GLSL("float angle = floor(theta * %d.0 / %s);" % (self.quant_angle, math.pi))
        GLSL("float strength = %s;" % quantize("lambda", self.min_strength, 0, self.quant_strength - 1))
        GLSL("float coherence = %s;" % quantize("mu", self.min_coherence, 0, self.quant_coherence - 1))

    def apply_convolution_kernel(self, samples_list):
        GLSL = self.add_glsl
        n = self.radius * 2 - 1

        GLSL("float coord_y = ((angle * %d.0 + strength) * %d.0 + coherence + 0.5) / %d.0;" %
             (self.quant_strength, self.quant_coherence, self.quant_angle * self.quant_strength * self.quant_coherence))

        GLSL("vec4 res0 = vec4(0.0), res1 = vec4(0.0), w0, w1;")

        for i in range(self.lut_width // 2):
            for j in range(2):
                coord_x = (float(i * 2 + j) + 0.5) / float(self.lut_width)
                GLSL("w%d = texture(%s, vec2(%s, coord_y));" % (j, self.lut_name, coord_x))
            j = n * n - 1 - i
            if i < j:
                GLSL("res0 += %s * w0 + %s * w1.wzyx;" % (samples_list[i], samples_list[j]))
                GLSL("res1 += %s * w1 + %s * w0.wzyx;" % (samples_list[i], samples_list[j]))
            elif i == j:
                GLSL("res0 += %s * w0;" % (samples_list[i]))
                GLSL("res1 += %s * w1;" % (samples_list[i]))

        GLSL("res0 = clamp(res0, 0.0, 1.0);")
        GLSL("res1 = clamp(res1, 0.0, 1.0);")

    def generate(self, block_size):
        self.reset()
        GLSL = self.add_glsl
        n = self.radius * 2 - 1

        self.set_description("RAVU-3x (r%d)" % self.radius)

        block_width, block_height = block_size

        self.set_skippable(3, 3)
        self.set_transform(3, 3, 0.0, 0.0)
        self.bind_tex(self.lut_name)
        self.set_compute(block_width * 3, block_height * 3,
                         block_width, block_height)

        offset_base = -(self.radius - 1)
        array_size = block_width + n - 1, block_height + n - 1
        GLSL("shared float inp[%d];" % (array_size[0] * array_size[1]))

        GLSL("""
void hook() {""")

        # load all samples
        GLSL("ivec2 group_base = ivec2(gl_WorkGroupID) * ivec2(gl_WorkGroupSize);")
        GLSL("int local_pos = int(gl_LocalInvocationID.x) * %d + int(gl_LocalInvocationID.y);" % array_size[1])

        GLSL("""
for (int id = int(gl_LocalInvocationIndex); id < %d; id += int(gl_WorkGroupSize.x * gl_WorkGroupSize.y)) {""" % (array_size[0] * array_size[1]))

        GLSL("int x = id / %d, y = id %% %d;" % (array_size[1], array_size[1]))

        GLSL("inp[id] = HOOKED_tex(HOOKED_pt * vec2(float(group_base.x+x)+(%s), float(group_base.y+y)+(%s))).x;" %
             (offset_base + 0.5, offset_base + 0.5))

        GLSL("""
}""")

        GLSL("groupMemoryBarrier();")
        GLSL("barrier();")

        samples_list = []
        for dx in range(1 - self.radius, self.radius):
            for dy in range(1 - self.radius, self.radius):
                offset = (dx - offset_base) * array_size[1] + (dy - offset_base)
                samples_list.append("inp[local_pos + %d]" % offset)

        self.extract_key(samples_list)

        self.apply_convolution_kernel(samples_list)

        for i in range(9):
            pos = "ivec2(gl_GlobalInvocationID) * 3 + ivec2(%d, %d)" % (i / 3, i % 3)
            if i < 4:
                output = "res0[%d]" % i
            elif i == 4:
                output = samples_list[len(samples_list) // 2]
            else:
                output = "res1[%d]" % (i - 5)
            GLSL("imageStore(out_image, %s, vec4(%s, 0.0, 0.0, 0.0));" % (pos, output))

        GLSL("""
}""")

        return super().generate()


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="generate RAVU-3x user shader for mpv")
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
        default=[None],
        help='allowed downscaling ratio (default: no limit)')
    parser.add_argument(
        '--compute-shader-block-size',
        nargs=2,
        metavar=('block_width', 'block_height'),
        default=[32, 8],
        type=int,
        help='specify the block size of compute shader (default: 32 8)')
    parser.add_argument(
        '--float-format',
        nargs=1,
        choices=FloatFormat.__members__,
        default=["float32"],
        help="specify the float format of LUT")

    args = parser.parse_args()
    weights_file = args.weights_file[0]
    max_downscaling_ratio = args.max_downscaling_ratio[0]
    compute_shader_block_size = args.compute_shader_block_size
    float_format = FloatFormat[args.float_format[0]]

    gen = RAVU_3x(hook=["LUMA"],
                  weights_file=weights_file,
                  target_tex="OUTPUT",
                  max_downscaling_ratio=max_downscaling_ratio)

    sys.stdout.write(userhook.LICENSE_HEADER)
    sys.stdout.write(gen.generate(compute_shader_block_size))
    sys.stdout.write(gen.generate_tex(float_format))
