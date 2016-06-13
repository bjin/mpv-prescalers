import enum

import userhook

#
# *******  Super XBR Shader  *******
#
# Copyright (c) 2015 Hyllian - sergiogdb@gmail.com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#


def _clamp(x, lo, hi):
    return min(max(x, lo), hi)


class Option:
    def __init__(self, sharpness=1.0, edge_strength=0.6):
        self.sharpness = _clamp(sharpness, 0.0, 2.0)
        self.edge_strength = _clamp(edge_strength, 0.0, 1.0)


class StepParam:
    def __init__(self, dstr, ostr, d1, d2, o1, o2):
        self.dstr, self.ostr = dstr, ostr  # sharpness strength modifiers
        self.d1 = d1  # 1-distance diagonal mask
        self.d2 = d2  # 2-distance diagonal mask
        self.o1 = o1  # 1-distance orthogonal mask
        self.o2 = o2  # 2-distance orthogonal mask


class Step(enum.Enum):
    step1 = StepParam(dstr=0.129633,
                      ostr=0.175068,
                      d1=[[0, 1, 0], [1, 2, 1], [0, 1, 0]],
                      d2=[[-1, 0], [0, -1]],
                      o1=[1, 2, 1],
                      o2=[0, 0])
    step2 = StepParam(dstr=0.175068,
                      ostr=0.129633,
                      d1=[[0, 1, 0], [1, 4, 1], [0, 1, 0]],
                      d2=[[0, 0], [0, 0]],
                      o1=[1, 4, 1],
                      o2=[0, 0])


class Target(enum.Enum):
    luma = 0
    rgb = 1


class SuperxBR(userhook.UserHook):
    def __init__(self, target=Target.luma, option=Option(), **args):
        super().__init__(**args)

        self.target = target
        self.option = option

    def _step_h(self, mask):
        # Compute a single step of the superxbr process, assuming the input
        # can be sampled using i(x,y). Dumps its output into 'res'
        GLSL = self.add_glsl

        GLSL('{ // step')

        GLSL("""
$sample4_type d1 = $sample4_type( i(0,0), i(1,1), i(2,2), i(3,3) );
$sample4_type d2 = $sample4_type( i(0,3), i(1,2), i(2,1), i(3,0) );
$sample4_type h1 = $sample4_type( i(0,1), i(1,1), i(2,1), i(3,1) );
$sample4_type h2 = $sample4_type( i(0,2), i(1,2), i(2,2), i(3,2) );
$sample4_type v1 = $sample4_type( i(1,0), i(1,1), i(1,2), i(1,3) );
$sample4_type v2 = $sample4_type( i(2,0), i(2,1), i(2,2), i(2,3) );""")

        GLSL('float dw = %f;' % (self.option.sharpness * mask.dstr))
        GLSL('float ow = %f;' % (self.option.sharpness * mask.ostr))
        GLSL('vec4 dk = vec4(-dw, dw+0.5, dw+0.5, -dw);')  # diagonal kernel
        GLSL('vec4 ok = vec4(-ow, ow+0.5, ow+0.5, -ow);')  # ortho kernel

        # Convoluted results
        GLSL('$sample_type d1c = SAMPLE4_MUL(d1, dk);')
        GLSL('$sample_type d2c = SAMPLE4_MUL(d2, dk);')
        GLSL('$sample_type vc = SAMPLE4_MUL(v1+v2, ok)/2.0;')
        GLSL('$sample_type hc = SAMPLE4_MUL(h1+h2, ok)/2.0;')

        # Compute diagonal edge strength using diagonal mask
        GLSL('float d_edge = 0.0;')
        for x in range(3):
            for y in range(3):
                if mask.d1[x][y]:
                    # 1-distance diagonal neighbours
                    GLSL('d_edge += %d.0 * abs(luma(%d,%d) - luma(%d,%d));' %
                         (mask.d1[x][y], x + 1, y, x, y + 1))
                    GLSL('d_edge -= %d.0 * abs(luma(%d,%d) - luma(%d,%d));' %
                         (mask.d1[x][y], 3 - y, x + 1, 3 - (y + 1), x))
                if x < 2 and y < 2 and mask.d2[x][y]:
                    # 2-distance diagonal neighbours
                    GLSL('d_edge += %d.0 * abs(luma(%d,%d) - luma(%d,%d));' %
                         (mask.d2[x][y], x + 2, y, x, y + 2))
                    GLSL('d_edge -= %d.0 * abs(luma(%d,%d) - luma(%d,%d));' %
                         (mask.d2[x][y], 3 - y, x + 2, 3 - (y + 2), x))

        # Compute orthogonal edge strength using orthogonal mask
        GLSL('float o_edge = 0.0;')
        for x in range(1, 3):
            for y in range(3):
                if mask.o1[y]:
                    # 1-distance neighbours
                    GLSL('o_edge += %d.0 * abs(luma(%d,%d) - luma(%d,%d));' %
                         (mask.o1[y], x, y, x, y + 1))  # vertical
                    GLSL('o_edge -= %d.0 * abs(luma(%d,%d) - luma(%d,%d));' %
                         (mask.o1[y], y, x, y + 1, x))  # horizontal
                if y < 2 and mask.o2[y]:
                    # 2-distance neighbours
                    GLSL('o_edge += %d.0 * abs(luma(%d,%d) - luma(%d,%d));' %
                         (mask.o2[y], x, y, x, y + 2))  # vertical
                    GLSL('o_edge -= %d.0 * abs(luma(%d,%d) - luma(%d,%d));' %
                         (mask.o2[x], y, x, y + 2, x))  # horizontal

        # Pick the two best directions and mix them together
        GLSL('float str = smoothstep(0.0, %f + 1e-6, abs(d_edge));' %
             self.option.edge_strength)
        GLSL("""
res = mix(mix(d2c, d1c, step(0.0, d_edge)),
      mix(hc,   vc, step(0.0, o_edge)), 1.0 - str);""")

        # Anti-ringing using center square
        GLSL("""
$sample_type lo = min(min( i(1,1), i(2,1) ), min( i(1,2), i(2,2) ));
$sample_type hi = max(max( i(1,1), i(2,1) ), max( i(1,2), i(2,2) ));
res = clamp(res, lo, hi);""")

        GLSL('} // step')

    def generate(self, step):
        self.reset()
        GLSL = self.add_glsl

        if self.target == Target.luma:
            self.add_mappings(sample_type="float",
                              sample_zero="0.0",
                              sample4_type="vec4",
                              function_args="int comp")
        else:
            self.add_mappings(sample_type="vec4",
                              sample_zero="vec4(0.0)",
                              sample4_type="mat4",
                              function_args="")
            # Assumes Rec. 709
            self.add_mappings(color_primary="vec4(0.2126, 0.7152, 0.0722, 0)")

        GLSL("""
$sample_type superxbr($function_args) {
$sample_type i[4*4];
$sample_type res;
#define i(x,y) i[(x)*4+(y)]""")

        if self.target == Target.luma:
            GLSL('#define luma(x, y) i((x), (y))')
            GLSL('#define GET_SAMPLE(pos) HOOKED_texOff(pos)[comp]')
            GLSL('#define SAMPLE4_MUL(sample4, w) dot((sample4), (w))')
        else:
            GLSL('float luma[4*4];')
            GLSL('#define luma(x, y) luma[(x)*4+(y)]')
            GLSL('#define GET_SAMPLE(pos) HOOKED_texOff(pos)')
            # samples are stored in columns, use right multiplication.
            GLSL('#define SAMPLE4_MUL(sample4, w) ((sample4)*(w))')

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

            GLSL('#define IDX(x, y) vec2(float(x)-1.25, float(y)-1.25)')

        else:
            # This is the second pass, so it will never be rotated
            GLSL("""
vec2 dir = fract(HOOKED_pos * HOOKED_size / 2.0) - 0.5;
if (dir.x * dir.y > 0.0)
    return GET_SAMPLE(0);""")

            GLSL('#define IDX(x, y) vec2(x+y-3,y-x)')

        # Load the input samples
        GLSL('for (int x = 0; x < 4; x++)')
        GLSL('for (int y = 0; y < 4; y++) {')
        GLSL('i(x,y) = GET_SAMPLE(IDX(x,y));')
        if self.target != Target.luma:
            GLSL('luma(x,y) = dot(i(x,y), $color_primary);')
        GLSL('}')

        self._step_h(step.value)

        GLSL("""
return res;
}  // superxbr""")

        comps = self.max_components()
        if self.target == Target.luma:
            args = ", ".join("superxbr(%d)" % i if i < comps else "0.0"
                             for i in range(4))
            GLSL("""
vec4 hook() {
    return vec4(%s);
}""" % args)
        else:
            GLSL("""
vec4 hook() {
    return superxbr();
}""")

        return super().generate()
