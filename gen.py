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

import nnedi3
import superxbr
import userhook


def run(hooks, suffix):
    gen = superxbr.SuperxBR(hook=hooks)
    with open("superxbr%s.hook" % suffix, "w") as f:
        f.write(userhook.LICENSE_HEADER)
        for step in list(superxbr.Step):
            f.write(gen.generate(step))
    for nns in list(nnedi3.Neurons):
        for window in list(nnedi3.Window):
            target_tex = "LUMA" if hooks == ["CHROMA"] else "OUTPUT"
            gen = nnedi3.NNEDI3(nns,
                                window,
                                hook=hooks,
                                target_tex=target_tex,
                                max_downscaling_ratio=1.6)

            filename = "nnedi3-%s-%s%s.hook" % (nns.name, window.name, suffix)

            with open(filename, "w") as f:
                f.write(userhook.LICENSE_HEADER)
                for step in list(nnedi3.Step):
                    f.write(gen.generate(step))


run(["LUMA"], "")
run(["CHROMA"], "-chroma")
run(["LUMA", "CHROMA"], "-yuv")
run(["LUMA", "CHROMA", "RGB", "XYZ"], "-all")

gen = superxbr.SuperxBR(hook=["MAIN"], target=superxbr.Target.rgb)
with open("superxbr-native.hook", "w") as f:
    f.write(userhook.LICENSE_HEADER)
    for step in list(superxbr.Step):
        f.write(gen.generate(step))
