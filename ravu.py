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
    step3 = 2
    step4 = 3


class Profile(enum.Enum):
    luma = 0
    rgb = 1
    yuv = 2


class RAVU(userhook.UserHook):
    """
    An experimental prescaler inspired by RAISR (Rapid and Accurate Image Super
    Resolution).
    """

    def __init__(self,
                 profile=Profile.luma,
                 weights_file=None,
                 lut_name="ravu_lut",
                 int_tex_name="ravu_int",
                 **args):
        super().__init__(**args)

        self.profile = profile
        self.lut_name = lut_name

        self.tex_name = [["HOOKED", int_tex_name + "01"],
                         [int_tex_name + "10", int_tex_name + "11"]]

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

    def generate_tex(self):
        import struct

        height = self.quant_angle * self.quant_strength * self.quant_coherence
        width = self.radius * self.radius

        weights = []
        for i in range(self.quant_angle):
            for j in range(self.quant_strength):
                for k in range(self.quant_coherence):
                    assert len(self.model_weights[i][j][k]) == width * 4
                    weights.extend(self.model_weights[i][j][k])
        weights_raw = struct.pack('<%df' % len(weights), *weights).hex()

        headers = [
            "//!TEXTURE %s" % self.lut_name,
            "//!SIZE %d %d" % (width, height),
            "//!COMPONENTS 4",
            "//!FORMAT 32f",
            "//!FILTER NEAREST"
        ]

        return "\n".join(headers + [weights_raw, ""])

    def get_sample_positions(self, offset, use_gather=False):
        n = self.radius * 2

        if offset == (1, 1):
            pos_func = lambda x, y: (self.tex_name[0][0], x - (n // 2 - 1), y - (n // 2 - 1))
        elif offset == (0, 1) or offset == (1, 0):
            def pos_func(x, y):
                x, y = x + y - (n - 1), y - x
                x += offset[0]
                y += offset[1]
                assert x % 2 == y % 2
                return (self.tex_name[x % 2][y % 2], x // 2, y // 2)
        else:
            raise Exception("invalid offset")

        sample_positions = {}
        for i in range(n):
            for j in range(n):
                tex, x, y = pos_func(i, j)
                # tex_name, tex_offset -> logical offset
                sample_positions.setdefault(tex, {})[x, y] = i, j

        gathered_positions = {}
        if use_gather:
            gather_offsets = [(0, 1), (1, 1), (1, 0), (0, 0)]
            for tex in sorted(sample_positions.keys()):
                mapping = sample_positions[tex]
                used_keys = set()
                for x, y in sorted(mapping.keys()):
                    # (x, y) should be the minimum among |tex_offsets|
                    tex_offsets = [(x + dx, y + dy) for dx, dy in gather_offsets]
                    if all(key in mapping and key not in used_keys for key in tex_offsets):
                        used_keys |= set(tex_offsets)
                        logical_offsets = [mapping[key] for key in tex_offsets]
                        # tex_name, tex_offset_base -> logical offset
                        gathered_positions.setdefault(tex, {})[x, y] = logical_offsets
                for key in used_keys:
                    del mapping[key]

        return sample_positions, gathered_positions

    def setup_profile(self):
        GLSL = self.add_glsl

        if self.profile == Profile.luma:
            self.add_mappings(
                sample_type="float",
                sample_zero="0.0",
                hook_return_value="vec4(res, 0.0, 0.0, 0.0)",
                comps_swizzle = "[0]")
        else:
            self.add_mappings(
                sample_type="vec4",
                sample_zero="vec4(0.0)",
                hook_return_value="res",
                comps_swizzle = "")
            if self.profile == Profile.rgb:
                # Assumes Rec. 709
                GLSL("const vec4 color_primary = vec4(0.2126, 0.7152, 0.0722, 0.0);")
            elif self.profile == Profile.yuv:
                # Add some no-op cond to assert LUMA texture exists, rather make
                # the shader failed to run than getting some random output.
                self.add_cond("LUMA.w 0 >")

    def extract_key(self, luma):
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
                    if x == 1 or x == n - 2:
                        return "(%s-%s)/2.0" % (f(x + 1), f(x - 1))
                    return "(-%s+8.0*%s-8.0*%s+%s)/12.0" % (f(x + 2), f(x + 1), f(x - 1), f(x - 2))

                GLSL("gx = %s;" % numerial_differential(
                    lambda i2: luma(i2, j), i))
                GLSL("gy = %s;" % numerial_differential(
                    lambda j2: luma(i, j2), j))
                gw = self.gaussian[i - gradient_left][j - gradient_left]
                GLSL("abd += vec3(gx * gx, gx * gy, gy * gy) * %s;" % gw)

        # Eigenanalysis of gradient matrix
        eps = "1e-9"
        GLSL("""
float a = abd.x, b = abd.y, d = abd.z;
float T = a + d, D = a * d - b * b;
float delta = sqrt(max(T * T / 4 - D, 0.0));
float L1 = T / 2 + delta, L2 = T / 2 - delta;
float V1x = b, V1y = L1 - a;
if (abs(b) < %s) { V1x = 1.0; V1y = 0.0; }
float sqrtL1 = sqrt(L1), sqrtL2 = sqrt(L2);
float theta = mod(atan(V1y, V1x) + %s, %s);
float lambda = sqrtL1;
float mu = mix((sqrtL1 - sqrtL2) / (sqrtL1 + sqrtL2), 0.0, (sqrtL1 + sqrtL2) < %s);
""" % (eps, math.pi, math.pi, eps))

        # Extract convolution kernel based on quantization of (angle, strength, coherence)
        GLSL("float angle = floor(theta * %d.0 / %s);" % (self.quant_angle, math.pi))
        GLSL("float strength, coherence;")

        def quantize(target_name, var_name, seps, l, r):
            if l == r:
                GLSL("%s = %d.0;\n" % (target_name, l))
                return
            m = (l + r) // 2
            GLSL("if (%s < %s) {" % (var_name, seps[m]))
            quantize(target_name, var_name, seps, l, m)
            GLSL("} else {")
            quantize(target_name, var_name, seps, m + 1, r)
            GLSL("}")

        quantize("strength", "lambda", self.min_strength, 0, self.quant_strength - 1)
        quantize("coherence", "mu", self.min_coherence, 0, self.quant_coherence - 1)

    def apply_convolution_kernel(self, samples_list):
        GLSL = self.add_glsl
        n = self.radius * 2

        assert len(samples_list) == n * n

        GLSL("float coord_y = ((angle * %d.0 + strength) * %d.0 + coherence + 0.5) / %d.0;" %
             (self.quant_strength, self.quant_coherence, self.quant_angle * self.quant_strength * self.quant_coherence))

        GLSL("$sample_type res = $sample_zero;")
        GLSL("vec4 w;")
        blocks = n * n // 4
        for i in range(blocks):
            coord_x = (float(i) + 0.5) / float(blocks)
            GLSL("w = texture(%s, vec2(%s, coord_y));" % (self.lut_name, coord_x))
            for j in range(4):
                GLSL("res += %s * w[%d];" % (samples_list[i * 4 + j], j))
        GLSL("res = clamp(res, 0.0, 1.0);")


    def generate(self, step, use_gather=False):
        self.reset()
        GLSL = self.add_glsl

        self.set_description("RAVU (%s, %s, r%d)" %
                             (step.name, self.profile.name, self.radius))

        # This checks against all passes, and works since "HOOKED" is same for
        # all of them.
        self.set_skippable(2, 2)

        if step == Step.step4:
            self.set_transform(2, 2, -0.5, -0.5)

            self.bind_tex(self.tex_name[0][1])
            self.bind_tex(self.tex_name[1][0])
            self.bind_tex(self.tex_name[1][1])

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
""" % (self.tex_name[0][0], self.tex_name[0][1],
       self.tex_name[1][0], self.tex_name[1][1]))

            return super().generate()

        self.bind_tex(self.lut_name)

        self.setup_profile()

        n = self.radius * 2
        samples = {(x, y): "sample%d" % (x * n + y) for x in range(n) for y in range(n)}

        if self.profile == Profile.luma:
            luma = lambda x, y: samples[x, y]
        elif self.profile == Profile.rgb:
            luma = lambda x, y: "luma%d" % (x * n + y)
        elif self.profile == Profile.yuv:
            luma = lambda x, y: samples[x, y] + "[0]"

        if step == Step.step1:
            offset = (1, 1)
        else:
            self.bind_tex(self.tex_name[1][1])
            if step == Step.step2:
                offset = (1, 0)
            elif step == Step.step3:
                offset = (0, 1)

        self.save_tex(self.tex_name[offset[0]][offset[1]])

        sample_positions, gathered_positions = self.get_sample_positions(
                offset, use_gather and self.profile == Profile.luma)

        gathered = 0
        for tex in sorted(gathered_positions.keys()):
            mapping = gathered_positions[tex]
            for base_x, base_y in sorted(mapping.keys()):
                logical_offsets = mapping[base_x, base_y]
                gathered_name = "gathered%d" % gathered
                gathered += 1
                GLSL("vec4 %s = %s_mul * textureGatherOffset(%s_raw, %s_pos, ivec2(%d, %d), 0);" %
                     (gathered_name, tex, tex, tex, base_x, base_y))
                for idx in range(len(logical_offsets)):
                    i, j = logical_offsets[idx]
                    samples[i, j] = "%s[%d]" % (gathered_name, idx)

        for tex in sorted(sample_positions.keys()):
            mapping = sample_positions[tex]
            for x, y in sorted(mapping.keys()):
                i, j = mapping[x, y]
                GLSL('$sample_type %s = %s_texOff(vec2(%d.0, %d.0))$comps_swizzle;' %
                     (samples[i, j], tex, x, y))
                if self.profile == Profile.rgb:
                    GLSL('float %s = dot(%s, color_primary);' % (luma(i, j), samples[i, j]))

        GLSL("""
vec4 hook() {""")

        self.extract_key(luma)

        samples_list = [samples[i, j] for i in range(n) for j in range(n)]
        self.apply_convolution_kernel(samples_list)

        GLSL("""
return $hook_return_value;
}""")

        return super().generate()

    def generate_compute(self, step, block_size):
        # compute shader requires only two steps
        if step == Step.step3 or step == Step.step4:
            return ""

        self.reset()
        GLSL = self.add_glsl

        self.set_description("RAVU (%s, %s, r%d, compute)" %
                             (step.name, self.profile.name, self.radius))

        block_width, block_height = block_size

        self.set_skippable(2, 2)
        self.bind_tex(self.lut_name)

        self.setup_profile()

        if step == Step.step1:
            self.set_compute(block_width, block_height)

            target_offsets = [(1, 1)]
            self.save_tex(self.tex_name[1][1])
        elif step == Step.step2:
            self.set_compute(block_width * 2, block_height * 2,
                             block_width, block_height)
            self.set_transform(2, 2, -0.5, -0.5)

            target_offsets = [(0, 1), (1, 0)]
            self.bind_tex(self.tex_name[1][1])

        n = self.radius * 2
        sample_positions_by_target = [self.get_sample_positions(target_offset, False)[0] for target_offset in target_offsets]

        # for each bound texture, declare global variables/shared arrays and
        # prepare index/samples mapping
        bound_tex_names = list(sample_positions_by_target[0].keys())
        offset_for_tex = []
        array_size_for_tex = []
        samples_mapping_for_target = [{} for sample_positions in sample_positions_by_target]
        for tex_idx, tex in enumerate(bound_tex_names):
            tex_offsets = set()
            for sample_positions in sample_positions_by_target:
                tex_offsets |= set(sample_positions[tex].keys())
            minx = min(key[0] for key in tex_offsets)
            maxx = max(key[0] for key in tex_offsets)
            miny = min(key[1] for key in tex_offsets)
            maxy = max(key[1] for key in tex_offsets)

            offset_for_tex.append((minx, miny))
            array_size = (maxx - minx + block_width, maxy - miny + block_height)
            array_size_for_tex.append(array_size)

            GLSL("shared $sample_type inp%d[%d][%d];" % (tex_idx, array_size[1], array_size[0]))
            if self.profile != Profile.luma:
                GLSL("shared float inp_luma%d[%d][%d];" % (tex_idx, array_size[1], array_size[0]))

            # Samples mapping are different for different sample_positions
            for target_idx, sample_positions in enumerate(sample_positions_by_target):
                samples_mapping = samples_mapping_for_target[target_idx]
                mapping = sample_positions[tex]
                for tex_offset in mapping.keys():
                    logical_offset = mapping[tex_offset]
                    samples_mapping[logical_offset] = "inp%d[gl_LocalInvocationID.y+%d][gl_LocalInvocationID.x+%d]" % \
                                                      (tex_idx, tex_offset[1] - miny, tex_offset[0] - minx)

        GLSL("""
void hook() {""")

        # load all samples
        GLSL("ivec2 group_base = ivec2(gl_WorkGroupID) * ivec2(gl_WorkGroupSize);")
        for tex_idx, tex in enumerate(bound_tex_names):
            offset_base = offset_for_tex[tex_idx]
            array_size = array_size_for_tex[tex_idx]
            GLSL("""
for (uint y = gl_LocalInvocationID.y; y < %d; y += gl_WorkGroupSize.y)
for (uint x = gl_LocalInvocationID.x; x < %d; x += gl_WorkGroupSize.x) {""" % (array_size[1], array_size[0]))

            GLSL("inp%d[y][x] = %s_mul * texelFetch(%s_raw, group_base + ivec2(x+(%d),y+(%d)), 0)$comps_swizzle;" %
                 (tex_idx, tex, tex, offset_base[0], offset_base[1]))

            if self.profile == Profile.yuv:
                GLSL("inp_luma%d[y][x] = inp%d[y][x][0];" % (tex_idx, tex_idx))
            elif self.profile == Profile.rgb:
                GLSL("inp_luma%d[y][x] = dot(inp%d[y][x], color_primary);" % (tex_idx, tex_idx))

            GLSL("""
}""")

        GLSL("groupMemoryBarrier();")
        GLSL("barrier();")

        for target_idx, sample_positions in enumerate(sample_positions_by_target):
            offset = target_offsets[target_idx]
            samples_mapping = samples_mapping_for_target[target_idx]

            GLSL("{")

            luma = lambda x, y: "luma%d" % (x * n + y)
            for sample_xy, (x, y) in sorted((samples_mapping[key], key) for key in samples_mapping.keys()):
                luma_xy = luma(x, y)
                if self.profile == Profile.luma:
                    GLSL("float %s = %s;" % (luma_xy, sample_xy))
                else:
                    GLSL("float %s = %s;" % (luma_xy, sample_xy.replace("inp", "inp_luma")))

            self.extract_key(luma)

            samples_list = [samples_mapping[i, j] for i in range(n) for j in range(n)]
            self.apply_convolution_kernel(samples_list)

            if step == Step.step1:
                pos = "ivec2(gl_GlobalInvocationID)"
            else:
                pos = "ivec2(gl_GlobalInvocationID) * 2 + ivec2(%d, %d)" % (offset[0], offset[1])
            GLSL("imageStore(out_image, %s, $hook_return_value);" % pos)

            GLSL("}")

        if step == Step.step2:
            GLSL("$sample_type res;")
            for tex_idx, tex in enumerate(bound_tex_names):
                offset_base = offset_for_tex[tex_idx]
                offset_global = 0 if tex == self.tex_name[0][0] else 1
                pos = "ivec2(gl_GlobalInvocationID) * 2 + ivec2(%d)" % offset_global
                res = "inp%d[gl_LocalInvocationID.y+%d][gl_LocalInvocationID.x+%d]" % (tex_idx, -offset_base[1], -offset_base[0])
                GLSL("res = %s;" % res)
                GLSL("imageStore(out_image, %s, $hook_return_value);" % pos)

        GLSL("""
}""")

        return super().generate()


if __name__ == "__main__":
    import argparse
    import sys

    profile_mapping = {
        "luma": (["LUMA"], Profile.luma),
        "native": (["MAIN"], Profile.rgb),
        "native-yuv": (["NATIVE"], Profile.yuv)
    }

    parser = argparse.ArgumentParser(
        description="generate RAVU user shader for mpv")
    parser.add_argument(
        '-t',
        '--target',
        nargs=1,
        choices=sorted(profile_mapping.keys()),
        default=["native"],
        help='target that shader is hooked on (default: native)')
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
        default=[32, 8],
        type=int,
        help='specify the block size of compute shader (default: 32 8)')

    args = parser.parse_args()
    target = args.target[0]
    hook, profile = profile_mapping[target]
    weights_file = args.weights_file[0]
    max_downscaling_ratio = args.max_downscaling_ratio[0]
    use_gather = args.use_gather
    use_compute_shader = args.use_compute_shader
    compute_shader_block_size = args.compute_shader_block_size

    gen = RAVU(hook=hook,
               profile=profile,
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
    sys.stdout.write(gen.generate_tex())
