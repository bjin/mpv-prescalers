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


class SuperxBR(userhook.UserHook):
    def __init__(self, option=Option(), **args):
        super().__init__(**args)

        self.option = option

    def _step_h(self, mask):
        # Compute a single step of the superxbr process, assuming the input
        # can be sampled using i(x,y). Dumps its output into 'res'
        GLSL = self.add_glsl

        GLSL('{ // step')

        GLSL("""
vec4 d1 = vec4( i(0,0), i(1,1), i(2,2), i(3,3) );
vec4 d2 = vec4( i(0,3), i(1,2), i(2,1), i(3,0) );
vec4 h1 = vec4( i(0,1), i(1,1), i(2,1), i(3,1) );
vec4 h2 = vec4( i(0,2), i(1,2), i(2,2), i(3,2) );
vec4 v1 = vec4( i(1,0), i(1,1), i(1,2), i(1,3) );
vec4 v2 = vec4( i(2,0), i(2,1), i(2,2), i(2,3) );""")

        GLSL('float dw = %f;' % (self.option.sharpness * mask.dstr))
        GLSL('float ow = %f;' % (self.option.sharpness * mask.ostr))
        GLSL('vec4 dk = vec4(-dw, dw+0.5, dw+0.5, -dw);')  # diagonal kernel
        GLSL('vec4 ok = vec4(-ow, ow+0.5, ow+0.5, -ow);')  # ortho kernel

        # Convoluted results
        GLSL('float d1c = dot(d1, dk);')
        GLSL('float d2c = dot(d2, dk);')
        GLSL('float vc = dot(v1+v2, ok)/2.0;')
        GLSL('float hc = dot(h1+h2, ok)/2.0;')

        # Compute diagonal edge strength using diagonal mask
        GLSL('float d_edge = 0.0;')
        for x in range(3):
            for y in range(3):
                if mask.d1[x][y]:
                    # 1-distance diagonal neighbours
                    GLSL('d_edge += %d.0 * abs(i(%d,%d) - i(%d,%d));' %
                         (mask.d1[x][y], x + 1, y, x, y + 1))
                    GLSL('d_edge -= %d.0 * abs(i(%d,%d) - i(%d,%d));' %
                         (mask.d1[x][y], 3 - y, x + 1, 3 - (y + 1), x))
                if x < 2 and y < 2 and mask.d2[x][y]:
                    # 2-distance diagonal neighbours
                    GLSL('d_edge += %d.0 * abs(i(%d,%d) - i(%d,%d));' %
                         (mask.d2[x][y], x + 2, y, x, y + 2))
                    GLSL('d_edge -= %d.0 * abs(i(%d,%d) - i(%d,%d));' %
                         (mask.d2[x][y], 3 - y, x + 2, 3 - (y + 2), x))

        # Compute orthogonal edge strength using orthogonal mask
        GLSL('float o_edge = 0.0;')
        for x in range(1, 3):
            for y in range(3):
                if mask.o1[y]:
                    # 1-distance neighbours
                    GLSL('o_edge += %d.0 * abs(i(%d,%d) - i(%d,%d));' %
                         (mask.o1[y], x, y, x, y + 1))  # vertical
                    GLSL('o_edge -= %d.0 * abs(i(%d,%d) - i(%d,%d));' %
                         (mask.o1[y], y, x, y + 1, x))  # horizontal
                if y < 2 and mask.o2[y]:
                    # 2-distance neighbours
                    GLSL('o_edge += %d.0 * abs(i(%d,%d) - i(%d,%d));' %
                         (mask.o2[y], x, y, x, y + 2))  # vertical
                    GLSL('o_edge -= %d.0 * abs(i(%d,%d) - i(%d,%d));' %
                         (mask.o2[x], y, x, y + 2, x))  # horizontal

        # Pick the two best directions and mix them together
        GLSL('float str = smoothstep(0.0, %f + 1e-6, abs(d_edge));' %
             self.option.edge_strength)
        GLSL("""
res = mix(mix(d2c, d1c, step(0.0, d_edge)),
      mix(hc,   vc, step(0.0, o_edge)), 1.0 - str);""")

        # Anti-ringing using center square
        GLSL("""
float lo = min(min( i(1,1), i(2,1) ), min( i(1,2), i(2,2) ));
float hi = max(max( i(1,1), i(2,1) ), max( i(1,2), i(2,2) ));
res = clamp(res, lo, hi);""")

        GLSL('} // step')

    def generate(self, step):
        self.clear_glsl()
        GLSL = self.add_glsl

        GLSL("""
float superxbr(int comp) {
float i[4*4];
float res;
#define i(x,y) i[(x)*4+(y)]""")

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
    return 0.0;""")

            GLSL("""
if (dir.x < 0.0 || dir.y < 0.0 || dist.x < 1.0 || dist.y < 1.0)
    return HOOKED_texOff(-dir)[comp];""")

            # Load the input samples
            GLSL("""
for (int x = 0; x < 4; x++)
for (int y = 0; y < 4; y++)
i(x,y) = HOOKED_texOff(vec2(float(x)-1.25, float(y)-1.25))[comp];""")
        else:
            # This is the second pass, so it will never be rotated
            GLSL("""
vec2 dir = fract(HOOKED_pos * HOOKED_size / 2.0) - 0.5;
if (dir.x * dir.y > 0.0)
    return HOOKED_texOff(0)[comp];""")
            GLSL("""
for (int x = 0; x < 4; x++)
for (int y = 0; y < 4; y++)
i(x,y) = HOOKED_texOff(vec2(x+y-3, y-x))[comp];""")

        self._step_h(step.value)

        GLSL("""
return res;
}  // superxbr""")

        comps = self.max_components()
        GLSL("""
vec4 hook() {
    return vec4(%s);
}""" % ", ".join("superxbr(%d)" % i if i < comps else "0.0" for i in range(4)))

        return super().generate()
