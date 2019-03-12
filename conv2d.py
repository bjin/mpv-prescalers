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

import numpy as np

import userhook

class Conv2D(userhook.UserHook):

    def __init__(self, input_channels, output_channels, kernel_size, **args):
        super().__init__(**args)

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size

    def generate(self, weights, bias=None, input_tex=None, output_tex=None):
        self.reset()
        GLSL = self.add_glsl

        self.set_description("Conv2D (%d, %d, %d)" %
                             (self.input_channels, self.output_channels, self.kernel_size))

        assert weights.shape == (self.input_channels, self.output_channels,
                                 self.kernel_size, self.kernel_size)
        assert bias is None or bias.shape == (self.output_channels,)

        # FIXME: generalize
        assert self.input_channels == 4
        assert self.output_channels == 4
        assert self.kernel_size % 2 == 1

        # FIXME: handle multiple textures, resized textures (for >4 feature channels)
        if input_tex is not None:
            self.bind_tex(input_tex)
            self.set_output_size("%s.w" % input_tex, "%s.h" % input_tex)
        if output_tex is not None:
            self.save_tex(output_tex)
        self.set_components(4)

        GLSL("vec4 hook() {")

        GLSL("vec4 res = vec4(0.0);")

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                ma = weights[...,i,j].T.ravel()
                x = i - self.kernel_size // 2
                y = j - self.kernel_size // 2
                GLSL("res += mat4(%s) * %s_texOff(vec2(%d.0, %d.0));" %
                     (",".join(repr(e) for e in ma), input_tex, x, y))

        if bias is not None:
            GLSL("res += vec4(%s);" % ",".join(repr(e) for e in bias))

        GLSL("return res;")

        GLSL("}")

        return super().generate()

class Conv2D_3x3(userhook.UserHook):

    def __init__(self, input_channels, output_channels, **args):
        super().__init__(**args)

        self.input_channels = input_channels
        self.output_channels = output_channels

    def generate(self, weights, bias=None, input_tex=None, output_tex=None, block_size=(32, 4)):
        self.reset()
        GLSL = self.add_glsl

        self.set_description("Conv2D_3x3 (%d, %d)" %
                             (self.input_channels, self.output_channels))

        assert weights.shape == (self.input_channels, self.output_channels, 3, 3)
        assert bias is None or bias.shape == (self.output_channels,)

        # FIXME: generalize
        assert self.input_channels == 4
        assert self.output_channels == 4

        group_size = 2
        self.set_compute(block_size[0] * group_size, block_size[1] * group_size,
                         block_size[0], block_size[1])

        # FIXME: handle multiple textures, resized textures (for >4 feature channels)
        if input_tex is not None:
            self.bind_tex(input_tex)
            self.set_output_size("%s.w" % input_tex, "%s.h" % input_tex)
        if output_tex is not None:
            self.save_tex(output_tex)
        self.set_components(4)

        # Fast Algorithms for Convolutional Neural Networks
        # https://arxiv.org/abs/1509.09308
        #
        # F(2, 3): Y = A.T [(G g G.T) * (B.T d B)] A
        B = np.array([(1,  0, -1,  0),
                      (0,  1,  1,  0),
                      (0, -1,  1,  0),
                      (0,  1,  0, -1)], dtype=int).T
        G = np.array([(1.0,  0.0, 0.0),
                      (0.5,  0.5, 0.5),
                      (0.5, -0.5, 0.5),
                      (0.0,  0.0, 1.0)], dtype=weights.dtype)
        A = np.array([(1, 1,  1,  0),
                      (0, 1, -1, -1)], dtype=int).T

        GgGT = G.dot(weights).dot(G.T).transpose((0, 3, 1, 2))
        BTiB = B.T.reshape((4, 4, 1)).dot(B.reshape((4, 1, 4))).transpose((0, 3, 1, 2))
        ATiA = A.T.reshape((2, 4, 1)).dot(A.reshape((4, 1, 2))).transpose((0, 3, 1, 2))

        array_size = block_size[0] * group_size + 2, block_size[1] * group_size + 2

        GLSL("shared vec4 inp[%d];" % (array_size[0] * array_size[1]))

        GLSL("void hook() {")

        GLSL("ivec2 group_base = ivec2(gl_WorkGroupID) * ivec2(gl_WorkGroupSize) * %d;" % group_size)
        GLSL("int local_pos = (int(gl_LocalInvocationID.x) * %d + int(gl_LocalInvocationID.y)) * %d;" % (array_size[1], group_size))

        GLSL("""
for (int id = int(gl_LocalInvocationIndex); id < %d; id += int(gl_WorkGroupSize.x * gl_WorkGroupSize.y)) {""" % (array_size[0] * array_size[1]))

        GLSL("int x = id / %d, y = id %% %d;" % (array_size[1], array_size[1]))

        GLSL("inp[id] = %s_tex(%s_pt * vec2(float(group_base.x + x) - 0.5, float(group_base.y + y) - 0.5));" % (input_tex, input_tex))

        GLSL("""
}""")

        GLSL("groupMemoryBarrier();")
        GLSL("barrier();")

        samples = {}
        for i in range(4):
            for j in range(4):
                offset = i * array_size[1] + j
                if i in [0, 3] and j in [0, 3]:
                    samples[i, j] = "inp[local_pos+%d]" % offset
                else:
                    samples[i, j] = "inp%d%d" % (i, j)
                    GLSL("vec4 %s = inp[local_pos+%d];" % (samples[i, j], offset))

        res_init = "vec4(%s)" % ",".join(repr(e) for e in bias) if bias is not None else "vec4(0.0)"

        res = {}
        for i in range(2):
            for j in range(2):
                res[i, j] = "res%d%d" % (i, j)
                GLSL("vec4 %s = %s;" % (res[i, j], res_init if i == 0 and j == 0 else res[0, 0]))

        GLSL("vec4 tmp;")
        for i in range(4):
            for j in range(4):
                sample_summed = ""
                for u in range(4):
                    for v in range(4):
                        val = BTiB[i, j, u, v]
                        if val != 0:
                            sample_summed += '+' if val > 0 else '-'
                            sample_summed += samples[u, v]

                sample_summed = sample_summed.lstrip('+')

                ma = "mat4(%s)" % ",".join(repr(e) for e in GgGT[i, j].T.ravel())

                GLSL("tmp = %s * (%s);" % (ma, sample_summed))

                lst = []
                for i0 in range(2):
                    for j0 in range(2):
                        val = ATiA[i0, j0, i, j]
                        if val != 0:
                            lst.append("%s %s= tmp;" % (res[i0, j0], "+" if val > 0 else "-"))
                GLSL(" ".join(lst))


        for i in range(2):
            for j in range(2):
                GLSL("imageStore(out_image, ivec2(gl_GlobalInvocationID) * %d + ivec2(%d, %d), %s);" % (group_size, i, j, res[i, j]))

        GLSL("}")

        return super().generate()

class Compare(userhook.UserHook):

    def __init__(self, **args):
        super().__init__(**args)

    def generate(self, input1, input2):
        self.reset()
        GLSL = self.add_glsl

        self.set_description("Compare (expecting all red)")

        self.bind_tex(input1)
        self.bind_tex(input2)

        GLSL("vec4 hook() {")

        GLSL("float diff = distance(%s_texOff(vec2(0.0)), %s_texOff(vec2(0.0)));" % (input1, input2))

        GLSL("""
if (diff < 0.0001) return vec4(1.0, 0.0, 0.0, 0.0);
if (diff < 0.01) return vec4(0.0, 1.0, 0.0, 0.0);
return vec4(0.0, 0.0, 1.0, 0.0);
        """)


        GLSL("}")

        return super().generate()


if __name__ == "__main__":
    w = np.random.random((4, 4, 3, 3)).astype(np.float32)
    for i in range(4):
        s = np.sum(w[:,i,...])
        w[:,i,...] /= s
        s = np.sum(w[:,i,...])
    b = (np.random.random((4,)) * 0.1 - 0.05).astype(np.float32)
    print(Conv2D(4, 4, 3, hook=["MAIN"]).generate(w, b, "HOOKED", "MODEL1"))
    print(Conv2D_3x3(4, 4, hook=["MAIN"]).generate(w, b, "HOOKED", "MODEL2"))
    print(Compare(hook=["MAIN"]).generate("MODEL1", "MODEL2"))
