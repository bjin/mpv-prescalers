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


class Profile(enum.Enum):
    luma = 0
    rgb = 1
    yuv = 2


class RAVU(userhook.UserHook):
    """
    An experimental prescaler inspired by RAISR (Rapid and Accurate Image Super
    Resolution).
    """

    def __init__(self, profile=Profile.luma, weights_file=None, **args):
        super().__init__(**args)

        self.profile = profile

        exec (open(weights_file).read())

        self.radius = locals()['radius']
        self.gradient_radius = locals()['gradient_radius']
        self.quant_angle = locals()['quant_angle']
        self.quant_strength = locals()['quant_strength']
        self.quant_coherence = locals()['quant_coherence']
        self.min_strength = locals()['min_strength']
        self.min_coherence = locals()['min_coherence']
        self.gaussian = locals()['gaussian']
        self.lr_weights = locals()['lr_weights']

        # XXX
        weights = []
        for h in range(self.radius * self.radius):
            for i in range(self.quant_angle):
                for j in range(self.quant_strength):
                    for k in range(self.quant_coherence):
                        weights.extend(
                            self.lr_weights[i][j][k][h * 4:h * 4 + 4])
        self.weights_bin = "/tmp/ravu.bin"
        import struct
        open(self.weights_bin,
             "wb").write(struct.pack('<%df' % len(weights), *weights))

    def generate(self, step):
        self.reset()
        GLSL = self.add_glsl

        # XXX
        GLSL('//!LOAD NEAREST %d %d 1 4 %s' %
             (self.quant_angle * self.quant_strength * self.quant_coherence,
              self.radius * self.radius, self.weights_bin))

        self.set_description(
            "RAVU (step=%s, profile=%s, radius=%d, gradient_radius=%d)" %
            (step.name, self.profile.name, self.radius, self.gradient_radius))

        if self.profile == Profile.luma:
            comps = self.max_components()
            args = ", ".join("ravu(%d)" % i if i < comps else "0.0"
                             for i in range(4))

            self.add_mappings(
                sample_type="float",
                sample_zero="0.0",
                function_args="int comp",
                hook_return_value="vec4(%s)" % args)
        else:
            self.add_mappings(
                sample_type="vec4",
                sample_zero="vec4(0.0)",
                function_args="",
                hook_return_value="ravu()")
            if self.profile == Profile.rgb:
                # Assumes Rec. 709
                self.add_mappings(
                    color_primary="vec4(0.2126, 0.7152, 0.0722, 0)")
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
            comps_suffix = "[comp]"
        else:
            if self.profile == Profile.rgb:
                luma = lambda x, y: "luma%d" % (x * n + y)
            elif self.profile == Profile.yuv:
                luma = lambda x, y: sample(x, y) + "[0]"
            comps_suffix = ""

        if step == Step.step1:
            self.set_transform(2, 2, -0.5, -0.5)
            GLSL("""
vec2 dir = fract(HOOKED_pos * HOOKED_size) - 0.5;
dir = transpose(HOOKED_rot) * dir;""")

            # Optimization: Discard (skip drawing) unused pixels, except those
            # at the edge.
            GLSL("""
vec2 dist = HOOKED_size * min(HOOKED_pos, vec2(1.0) - HOOKED_pos);
if (dir.x * dir.y < 0.0 && dist.x > 1.0 && dist.y > 1.0)
    return $sample_zero;""")

            GLSL("""
if (dir.x < 0.0 || dir.y < 0.0 || dist.x < 1.0 || dist.y < 1.0)
    return HOOKED_texOff(-dir)%s;""" % comps_suffix)

            get_position = lambda x, y: "vec2(%s,%s)" % (x - 0.25 - (n / 2 - 1), y - 0.25 - (n / 2 - 1))

        else:
            # This is the second pass, so it will never be rotated
            GLSL("""
vec2 dir = fract(HOOKED_pos * HOOKED_size / 2.0) - 0.5;
if (dir.x * dir.y > 0.0)
    return HOOKED_texOff(0)%s;""" % comps_suffix)

            get_position = lambda x, y: "vec2(%s,%s)" % (x + y - (n - 1), y - x)

        # Load the input samples
        for i in range(n):
            for j in range(n):
                GLSL('$sample_type %s = HOOKED_texOff(%s)%s;' %
                     (sample(i, j), get_position(i, j), comps_suffix))
                if self.profile == Profile.rgb:
                    GLSL('float %s = dot(%s, $color_primary);' %
                         (luma(i, j), sample(i, j)))

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
                    return "(-%s+8.0*%s-8.0*%s+%s)/12.0" % (f(x + 2), f(x + 1),
                                                            f(x - 1), f(x - 2))

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
        GLSL("float angle = floor(theta * %d.0 / %s);" % (self.quant_angle,
                                                          math.pi))
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

        quantize("strength", "lambda", self.min_strength, 0,
                 self.quant_strength - 1)
        quantize("coherence", "mu", self.min_coherence, 0,
                 self.quant_coherence - 1)
        GLSL(
            "float coord_x = ((angle * %d.0 + strength) * %d.0 + coherence + 0.5) / %d.0;"
            % (self.quant_strength, self.quant_coherence,
               self.quant_angle * self.quant_strength * self.quant_coherence))

        GLSL("$sample_type res = $sample_zero;")
        GLSL("vec4 w;")
        for i in range(n * n // 4):
            coord_y = (float(i) + 0.5) / float(n * n // 4)
            GLSL("w = texture(user_tex, vec2(coord_x, %s));" % coord_y)
            for j in range(4):
                GLSL("res += %s * w[%d];" % (samples[i * 4 + j], j))

        GLSL("""
return res;
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
        default=["luma"],
        help='target that shader is hooked on (default: luma)')
    parser.add_argument(
        '-w',
        '--weights-file',
        nargs=1,
        required=True,
        type=str,
        help='weights file name')

    args = parser.parse_args()
    target = args.target[0]
    hook = hooks[target]
    profile = native_profiles[
        target] if target in native_profiles else Profile.luma

    gen = RAVU(hook=hook, profile=profile, weights_file=args.weights_file[0])
    sys.stdout.write(userhook.LICENSE_HEADER)
    for step in list(Step):
        sys.stdout.write(gen.generate(step))
