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

        exec(open(weights_file).read())

        self.radius = locals()['radius']
        self.gradient_radius = locals()['gradient_radius']
        self.quant_angle = locals()['quant_angle']
        self.quant_strength = locals()['quant_strength']
        self.quant_coherence = locals()['quant_coherence']
        self.min_strength = locals()['min_strength']
        self.min_coherence = locals()['min_coherence']
        self.gaussian = locals()['gaussian']
        self.lr_weights = locals()['lr_weights']

    def generate(self, step):
        self.reset()
        GLSL = self.add_glsl

        self.set_description("RAVU (step=%s, profile=%s, radius=%d, gradient_radius=%d)" %
                             (step.name, self.profile.name, self.radius, self.gradient_radius))

        if self.profile == Profile.luma:
            comps = self.max_components()
            args = ", ".join("ravu(%d)" % i if i < comps else "0.0"
                             for i in range(4))

            self.add_mappings(sample_type="float",
                              sample_zero="0.0",
                              function_args="int comp",
                              hook_return_value="vec4(%s)" % args)
        else:
            self.add_mappings(sample_type="vec4",
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

        n = self.radius * 2

        GLSL("""
$sample_type ravu($function_args) {
$sample_type i[%d];
#define i(x,y) i[(x)*%d+(y)]""" % (n * n, n))

        if self.profile == Profile.luma:
            GLSL('#define luma(x, y) i((x), (y))')
            GLSL('#define GET_SAMPLE(pos) HOOKED_texOff(pos)[comp]')
        else:
            if self.profile == Profile.rgb:
                GLSL('float luma[%d];' % (n * n))
                GLSL('#define luma(x, y) luma[(x)*%d+(y)]' % n)
            elif self.profile == Profile.yuv:
                GLSL('#define luma(x, y) i(x,y)[0]')
            GLSL('#define GET_SAMPLE(pos) HOOKED_texOff(pos)')

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
    return GET_SAMPLE(-dir);""")

            GLSL('#define IDX(x, y) vec2(float(x)-0.25-%d.0,float(y)-0.25-%d.0)'
                    % (n / 2 - 1, n / 2 - 1))

        else:
            # This is the second pass, so it will never be rotated
            GLSL("""
vec2 dir = fract(HOOKED_pos * HOOKED_size / 2.0) - 0.5;
if (dir.x * dir.y > 0.0)
    return GET_SAMPLE(0);""")

            GLSL('#define IDX(x, y) vec2(x+y-%d.0,y-x)' % (n - 1))

        # Load the input samples
        GLSL('for (int x = 0; x < %d; x++)' % n)
        GLSL('for (int y = 0; y < %d; y++) {' % n)
        GLSL('i(x,y) = GET_SAMPLE(IDX(x,y));')
        if self.profile == Profile.rgb:
            GLSL('luma(x,y) = dot(i(x,y), $color_primary);')
        GLSL('}')

        # Calculate local gradient
        gradient_left = self.radius - self.gradient_radius
        gradient_right = n - gradient_left

        GLSL('float a = 0, b = 0, d = 0, gx, gy;')
        for i in range(gradient_left, gradient_right):
            for j in range(gradient_left, gradient_right):
                def numerial_differential(fx, x):
                    if x == 0:
                        return "(%s-%s)" % (fx % (x + 1), fx % x)
                    if x == n - 1:
                        return "(%s-%s)" % (fx % x, fx % (x - 1))
                    if x == 1 or x == n - 2:
                        return "(%s-%s)/2.0" % (fx % (x + 1), fx % (x - 1))
                    return "(-%s+8.0*%s-8.0*%s+%s)/12.0" % (fx % (x + 2), fx % (x + 1), fx % (x - 1), fx % (x - 2))
                GLSL("gx = %s;" % numerial_differential("luma(%%d, %d)" % j, i))
                GLSL("gy = %s;" % numerial_differential("luma(%d, %%d)" % i, j))
                gw = self.gaussian[i - gradient_left][j - gradient_left]
                GLSL("a += gx * gx * %s;" % gw);
                GLSL("b += gx * gy * %s;" % gw);
                GLSL("d += gy * gy * %s;" % gw);

        # Eigenanalysis of gradient matrix
        eps = "1e-9"
        pi = str(math.pi)
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
""" % (eps, pi, pi, eps))

        # Extract weights based on quantization of (angle, strength, coherence)
        GLSL("float ws[%d];" % (n * n))

        min_angle = [float(i) / float(self.quant_angle) * math.pi
                     for i in range(1, self.quant_angle)]

        def extract_weights(ws, *dims):
            if len(dims) == 0:
                GLSL("".join("ws[%d]=%s;" % (i, ws[i]) for i in range(len(ws))))
                return
            var_name, seps = dims[0]
            st = [(0, len(seps))]
            while len(st):
                e = st.pop()
                if isinstance(e, str):
                    GLSL(e)
                else:
                    l, r = e
                    if l == r:
                        extract_weights(ws[l], *dims[1:])
                    else:
                        mid = (l + r) // 2
                        st.extend([ "if (%s < %s) {" % (var_name, seps[mid])
                                  , (l, mid)
                                  , "} else {"
                                  , (mid + 1, r)
                                  , "}"
                                  ][::-1])

        extract_weights(self.lr_weights,
                        ("theta", min_angle),
                        ("lambda", self.min_strength),
                        ("mu", self.min_coherence))

        # Convolution kernel
        GLSL("""
$sample_type res = $sample_zero;
for (int x = 0; x < %d; x++)
res += i[x] * ws[x];
""" % (n * n))


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

    hooks = {"luma": ["LUMA"],
             "chroma": ["CHROMA"],
             "yuv": ["LUMA", "CHROMA"],
             "all": ["LUMA", "CHROMA", "RGB", "XYZ"],
             "native": ["MAIN"],
             "native-yuv": ["NATIVE"]}
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
    parser.add_argument('-w',
                        '--weights-file',
                        nargs=1,
                        required=True,
                        type=str,
                        help='weights file name')

    args = parser.parse_args()
    target = args.target[0]
    hook = hooks[target]
    profile = native_profiles[target] if target in native_profiles else Profile.luma

    gen = RAVU(hook=hook, profile=profile, weights_file = args.weights_file[0])
    sys.stdout.write(userhook.LICENSE_HEADER)
    for step in list(Step):
        sys.stdout.write(gen.generate(step))
