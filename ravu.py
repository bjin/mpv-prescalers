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
        self.int_tex_name = int_tex_name

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

    def generate(self, step, use_gather=False):
        self.reset()
        GLSL = self.add_glsl

        self.set_description("RAVU (%s, %s, r%d)" %
                             (step.name, self.profile.name, self.radius))

        self.bind_tex(self.lut_name)
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
            comps = self.max_components()
            if comps > 1:
                args = ", ".join("ravu(%d)" % i if i < comps else "0.0"
                                 for i in range(4))
                self.add_mappings(
                    sample_type="float",
                    sample_zero="0.0",
                    function_args="int comp",
                    hook_return_value="vec4(%s)" % args)
                comps_suffix = "[comp]"
            else:
                self.add_mappings(
                    sample_type="float",
                    sample_zero="0.0",
                    function_args="",
                    hook_return_value="vec4(ravu(), 0.0, 0.0, 0.0)")
                comps_suffix = "[0]"
        else:
            self.add_mappings(
                sample_type="vec4",
                sample_zero="vec4(0.0)",
                function_args="",
                hook_return_value="ravu()")
            comps_suffix = ""
            if self.profile == Profile.rgb:
                # Assumes Rec. 709
                GLSL("vec4 color_primary = vec4(0.2126, 0.7152, 0.0722, 0.0);")
            elif self.profile == Profile.yuv:
                # Add some no-op cond to assert LUMA texture exists, rather make
                # the shader failed to run than getting some random output.
                self.add_cond("LUMA.w 0 >")

        GLSL("""
$sample_type ravu($function_args) {""")

        n = self.radius * 2
        sample = lambda x, y: "sample%d" % (x * n + y)
        samples = [sample(i, j) for i in range(n) for j in range(n)]

        if self.profile == Profile.luma:
            luma = sample
        elif self.profile == Profile.rgb:
            luma = lambda x, y: "luma%d" % (x * n + y)
        elif self.profile == Profile.yuv:
            luma = lambda x, y: sample(x, y) + "[0]"

        if step == Step.step1:
            self.save_tex(tex_name[1][1])

            get_position = lambda x, y: (tex_name[0][0], x - (n // 2 - 1), y - (n // 2 - 1))
        else:
            self.bind_tex(tex_name[1][1])

            if step == Step.step2:
                offset_x, offset_y = 1, 0
            elif step == Step.step3:
                offset_x, offset_y = 0, 1

            self.save_tex(tex_name[offset_x][offset_y])

            def get_position(x, y):
                x, y = x + y - (n - 1), y - x
                x += offset_x
                y += offset_y
                assert x % 2 == y % 2
                return (tex_name[x % 2][y % 2], x // 2, y // 2)

        sample_positions = {}
        for i in range(n):
            for j in range(n):
                tex, x, y = get_position(i, j)
                sample_positions.setdefault(tex, {})[x, y] = i, j

        if use_gather and comps_suffix == "[0]":
            gather_offsets = [(0, 1), (1, 1), (1, 0), (0, 0)]
            GLSL("vec4 gathered;")
            for tex in sorted(sample_positions.keys()):
                mapping = sample_positions[tex]
                used_keys = set()
                for x, y in sorted(mapping.keys()):
                    local_group = [(x + dx, y + dy) for dx, dy in gather_offsets]
                    if all(key in mapping and key not in used_keys
                           for key in local_group):
                        used_keys |= set(local_group)
                        GLSL("gathered = %s_mul * textureGatherOffset(%s_raw, %s_pos, ivec2(%d, %d), 0);" % (tex, tex, tex, x, y))
                        for k, key in enumerate(local_group):
                            nx, ny = mapping[key]
                            GLSL('$sample_type %s = gathered[%d];' % (sample(nx, ny), k))
                for key in used_keys:
                    del mapping[key]

        for tex in sorted(sample_positions.keys()):
            mapping = sample_positions[tex]
            for base_x, base_y in sorted(mapping.keys()):
                i, j = mapping[base_x, base_y]
                GLSL('$sample_type %s = %s_texOff(vec2(%d.0, %d.0))%s;' %
                     (sample(i, j), tex, base_x, base_y, comps_suffix))
                if self.profile == Profile.rgb:
                    GLSL('float %s = dot(%s, color_primary);' % (luma(i, j), sample(i, j)))

        # Calculate local gradient
        gradient_left = self.radius - self.gradient_radius
        gradient_right = n - gradient_left

        GLSL('float a = 0, b = 0, d = 0, gx, gy;')
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
                GLSL("a += gx * gx * %s;" % gw)
                GLSL("b += gx * gy * %s;" % gw)
                GLSL("d += gy * gy * %s;" % gw)

        # Eigenanalysis of gradient matrix
        eps = "1e-9"
        GLSL("""
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
        GLSL("float coord_y = ((angle * %d.0 + strength) * %d.0 + coherence + 0.5) / %d.0;" %
             (self.quant_strength, self.quant_coherence, self.quant_angle * self.quant_strength * self.quant_coherence))

        GLSL("$sample_type res = $sample_zero;")
        GLSL("vec4 w;")
        blocks = n * n // 4
        for i in range(blocks):
            coord_x = (float(i) + 0.5) / float(blocks)
            GLSL("w = texture(%s, vec2(%s, coord_y));" % (self.lut_name, coord_x))
            for j in range(4):
                GLSL("res += %s * w[%d];" % (samples[i * 4 + j], j))

        GLSL("""
return clamp(res, 0.0, 1.0);
}  // ravu""")

        GLSL("""
vec4 hook() {
    return $hook_return_value;
}""")

        return super().generate()


if __name__ == "__main__":
    import argparse
    import sys

    hooks = {
        "luma": ["LUMA"],
        "chroma": ["CHROMA"],
        "yuv": ["LUMA", "CHROMA"],
        "all": ["LUMA", "CHROMA", "RGB", "XYZ"],
        "native": ["MAIN"],
        "native-yuv": ["NATIVE"]
    }
    native_profiles = {"native": Profile.rgb, "native-yuv": Profile.yuv}

    parser = argparse.ArgumentParser(
        description="generate RAVU user shader for mpv")
    parser.add_argument(
        '-t',
        '--target',
        nargs=1,
        choices=sorted(hooks.keys()),
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

    args = parser.parse_args()
    target = args.target[0]
    hook = hooks[target]
    profile = native_profiles.get(target, Profile.luma)
    weights_file = args.weights_file[0]
    max_downscaling_ratio = args.max_downscaling_ratio[0]
    use_gather = args.use_gather

    target_tex = "LUMA" if hook == ["CHROMA"] else "OUTPUT"
    gen = RAVU(hook=hook,
               profile=profile,
               weights_file=weights_file,
               target_tex=target_tex,
               max_downscaling_ratio=max_downscaling_ratio)

    sys.stdout.write(userhook.LICENSE_HEADER)
    for step in list(Step):
        sys.stdout.write(gen.generate(step, use_gather))
    sys.stdout.write(gen.generate_tex())
