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

class Step(enum.Enum):
    step1 = 0
    step2 = 1


class FloatFormat(enum.Enum):
    float16gl = 0
    float16vk = 1
    float32 = 2


class RAVU_Lite(userhook.UserHook):
    """
    A faster, slightly-lower-quality and luma-only variant of RAVU.
    """

    def __init__(self,
                 weights_file=None,
                 lut_name="ravu_lite_lut",
                 int_tex_name="ravu_lite_int",
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

        self.gather_offsets = [(0, 1), (1, 1), (1, 0), (0, 0)]
        self.prepare_positions()

    def prepare_positions(self):
        n = self.radius * 2

        self.gathered_groups = []
        self.gathered_group_base = []
        for i in range(0, n, 2):
            for j in range(0, n, 2):
                group = []
                for ox, oy in self.gather_offsets:
                    group.append((i + ox) * n + (j + oy))
                self.gathered_groups.append(group)
                self.gathered_group_base.append((i - self.radius + 1, j - self.radius + 1))
        self.luma = {}
        for i, gi in enumerate(self.gathered_groups):
            for j in range(4):
                self.luma[gi[j]] = "g%d.%s" % (i, "xyzw"[j])

    def generate_tex(self, float_format=FloatFormat.float32):
        import struct

        tex_format, item_format_str = {
            FloatFormat.float16gl: ("rgba16f", 'f'),
            FloatFormat.float16vk: ("rgba16hf", 'e'),
            FloatFormat.float32:   ("rgba32f", 'f')
        }[float_format]

        height = self.quant_angle * self.quant_strength * self.quant_coherence
        width = len(self.gathered_groups) * 3

        weights = []
        for i in range(self.quant_angle):
            for j in range(self.quant_strength):
                for k in range(self.quant_coherence):
                    for group in self.gathered_groups:
                        for z in range(3):
                            for idx in group:
                                weights.append(self.model_weights[i][j][k][z][idx])
        assert len(weights) == width * height * 4
        weights_raw = struct.pack('<%d%s' % (len(weights), item_format_str), *weights).hex()

        headers = [
            "//!TEXTURE %s" % self.lut_name,
            "//!SIZE %d %d" % (width, height),
            "//!FORMAT %s" % tex_format,
            "//!FILTER NEAREST"
        ]

        return "\n".join(headers + [weights_raw, ""])

    def extract_key(self):
        GLSL = self.add_glsl
        n = self.radius * 2

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
                    lambda i2: self.luma[i2 * n + j], i))
                GLSL("gy = %s;" % numerial_differential(
                    lambda j2: self.luma[i * n + j2], j))
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

    def apply_convolution_kernel(self):
        GLSL = self.add_glsl
        n = self.radius * 2

        GLSL("float coord_y = ((angle * %d.0 + strength) * %d.0 + coherence + 0.5) / %d.0;" %
             (self.quant_strength, self.quant_coherence, self.quant_angle * self.quant_strength * self.quant_coherence))

        GLSL("float r0 = 0.0, r1 = 0.0, r2 = 0.0;")
        blocks = len(self.gathered_groups) * 3
        for i, gi in enumerate(self.gathered_groups):
            for j in range(3):
                coord_x = (float(i * 3 + j) + 0.5) / float(blocks)
                GLSL("r%d += dot(g%d, texture(%s, vec2(%s, coord_y)));" % (j, i, self.lut_name, coord_x))

        GLSL("vec4 res = clamp(vec4(%s, r0, r1, r2), 0.0, 1.0);" % self.luma[(self.radius - 1) * n + (self.radius - 1)])

    def generate(self, step, use_gather=False):
        self.reset()
        GLSL = self.add_glsl
        n = self.radius * 2

        self.set_description("RAVU-Lite (%s, r%d)" % (step.name, self.radius))

        self.set_skippable(2, 2)

        if step == Step.step2:
            self.set_transform(2, 2, -0.5, -0.5)

            self.bind_tex(self.int_tex_name)

            GLSL("""
vec4 hook() {
    vec2 dir = fract(HOOKED_pos * HOOKED_size) - 0.5;
    int idx = int(dir.x > 0.0) * 2 + int(dir.y > 0.0);
    return vec4(%s_texOff(-dir)[idx], 0.0, 0.0, 0.0);
}
""" % self.int_tex_name)

            return super().generate()

        self.bind_tex(self.lut_name)
        self.save_tex(self.int_tex_name)
        self.set_components(4)

        GLSL("""
vec4 hook() {""")

        for i, gi in enumerate(self.gathered_groups):
            bx, by = self.gathered_group_base[i]
            if use_gather:
                GLSL("vec4 g%d = HOOKED_mul * textureGatherOffset(HOOKED_raw, HOOKED_pos, ivec2(%d, %d), 0);" % (i, bx, by))
            else:
                to_fetch = ["HOOKED_texOff(vec2(%d.0, %d.0)).x" % (bx + ox, by + oy) for ox, oy in self.gather_offsets]
                GLSL("vec4 g%d = vec4(%s);" % (i, ",".join(to_fetch)))

        self.extract_key()

        self.apply_convolution_kernel()

        GLSL("""
return res;
}""")

        return super().generate()

    def generate_compute(self, step, block_size):
        # compute shader requires only one step
        if step != Step.step1:
            return ""

        self.reset()
        GLSL = self.add_glsl
        n = self.radius * 2

        self.set_description("RAVU-Lite (r%d, compute)" % self.radius)

        block_width, block_height = block_size

        self.set_skippable(2, 2)
        self.set_transform(2, 2, -0.5, -0.5)
        self.bind_tex(self.lut_name)
        self.set_compute(block_width * 2, block_height * 2,
                         block_width, block_height)

        offset_base = -(self.radius - 1)
        array_size = block_width + n - 1, block_height + n - 1
        GLSL("shared float inp[%d][%d];" % (array_size[0], array_size[1]))

        GLSL("""
void hook() {""")

        # load all samples
        GLSL("ivec2 group_base = ivec2(gl_WorkGroupID) * ivec2(gl_WorkGroupSize);")

        GLSL("""
for (int x = int(gl_LocalInvocationID.x); x < %d; x += int(gl_WorkGroupSize.x))
for (int y = int(gl_LocalInvocationID.y); y < %d; y += int(gl_WorkGroupSize.y)) {""" % (array_size[0], array_size[1]))

        GLSL("inp[x][y] = HOOKED_tex(HOOKED_pt * vec2(float(group_base.x+x)+(%s), float(group_base.y+y)+(%s))).x;" %
             (offset_base + 0.5, offset_base + 0.5))

        GLSL("""
}""")

        GLSL("groupMemoryBarrier();")
        GLSL("barrier();")

        for i, gi in enumerate(self.gathered_groups):
            bx, by = self.gathered_group_base[i]
            to_fetch = ["inp[int(gl_LocalInvocationID.x)+%d][int(gl_LocalInvocationID.y)+%d]" %
                        (bx + ox - offset_base, by + oy - offset_base) for ox, oy in self.gather_offsets]
            GLSL("vec4 g%d = vec4(%s);" % (i, ",".join(to_fetch)))

        self.extract_key()

        self.apply_convolution_kernel()

        for i in range(4):
            pos = "ivec2(gl_GlobalInvocationID) * 2 + ivec2(%d, %d)" % (i / 2, i % 2)
            GLSL("imageStore(out_image, %s, vec4(res[%d], 0.0, 0.0, 0.0));" % (pos, i))

        GLSL("""
}""")

        return super().generate()


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="generate RAVU-Lite user shader for mpv")
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
        '--use-gather',
        action='store_true',
        help="enable use of textureGatherOffset (requires OpenGL 4.0)")
    parser.add_argument(
        '--use-compute-shader',
        action='store_true',
        help="enable use of compute shader (requires OpenGL 4.3)")
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
    use_gather = args.use_gather
    use_compute_shader = args.use_compute_shader
    compute_shader_block_size = args.compute_shader_block_size
    float_format = FloatFormat[args.float_format[0]]

    gen = RAVU_Lite(hook=["LUMA"],
                    weights_file=weights_file,
                    target_tex="OUTPUT",
                    max_downscaling_ratio=max_downscaling_ratio)

    sys.stdout.write(userhook.LICENSE_HEADER)
    for step in list(Step):
        if use_compute_shader:
            shader = gen.generate_compute(step, compute_shader_block_size)
        else:
            shader = gen.generate(step, use_gather)
        sys.stdout.write(shader)
    sys.stdout.write(gen.generate_tex(float_format))
