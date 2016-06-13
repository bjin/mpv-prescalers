"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from string import Template

HOOK = "HOOK"
BIND = "BIND"
SAVE = "SAVE"
WIDTH = "WIDTH"
HEIGHT = "HEIGHT"
OFFSET = "OFFSET"
WHEN = "WHEN"
COMPONENTS = "COMPONENTS"

HEADERS = [HOOK, BIND, SAVE, WIDTH, HEIGHT, OFFSET, WHEN, COMPONENTS]

HOOKED = "HOOKED"

LICENSE_HEADER = "".join(map("// {}\n".format, __doc__.splitlines())) + "\n"


class UserHook:
    def __init__(self,
                 hook=[],
                 bind=[],
                 save=[],
                 cond=None,
                 components=None,
                 target_tex=None,
                 max_downscaling_ratio=None):
        if HOOKED not in bind:
            bind.append(HOOKED)

        self.header = {}
        self.header[HOOK] = list(hook)
        self.header[BIND] = list(bind)
        self.header[SAVE] = list(save)
        if components:
            self.headers[COMPONENTS] = [str(components)]
        self.cond = cond
        self.target_tex = target_tex
        self.max_downscaling_ratio = max_downscaling_ratio

        self.reset()

    def add_glsl(self, line):
        self.glsl.append(line.strip())

    def reset(self):
        self.glsl = []
        self.header[WIDTH] = None
        self.header[HEIGHT] = None
        self.header[OFFSET] = None
        self.header[WHEN] = self.cond
        self.mappings = None

    def check_bind(self, used_tex):
        if used_hook not in self.header[BIND]:
            raise Exception('Texture %s is not binded' % used_tex)

    def add_mappings(self, **mappings):
        if self.mappings:
            self.mappings.update(mappings)
        else:
            self.mappings = mappings

    def add_cond(self, cond_str):
        if self.header[WHEN]:
            # Use boolean AND to apply multiple condition check.
            self.header[WHEN] = "%s %s *" % (self.header[WHEN], cond_str)
        else:
            self.header[WHEN] = cond_str

    def set_transform(self, mul_x, mul_y, offset_x, offset_y, skippable=False):
        if mul_x != 1:
            self.header[WIDTH] = "%d %s.w *" % (mul_x, HOOKED)
        if mul_y != 1:
            self.header[HEIGHT] = "%d %s.h *" % (mul_y, HOOKED)
        if skippable and self.target_tex and self.max_downscaling_ratio:
            # This step can be independently skipped, add WHEN condition.
            if mul_x > 1:
                self.add_cond(
                    "HOOKED.w %d * %s.w / %f <" %
                    (mul_x, self.target_tex, self.max_downscaling_ratio))
            if mul_y > 1:
                self.add_cond(
                    "HOOKED.h %d * %s.h / %f <" %
                    (mul_y, self.target_tex, self.max_downscaling_ratio))
        self.header[OFFSET] = ["%f %f" % (offset_x, offset_y)]

    def generate(self):
        headers = []
        for name in HEADERS:
            if name in self.header:
                value = self.header[name]
                if isinstance(value, list):
                    for arg in value:
                        headers.append("//!%s %s" % (name, arg.strip()))
                elif isinstance(value, str):
                    headers.append("//!%s %s" % (name, value.strip()))

        hook = "\n".join(headers + self.glsl + [""])
        if self.mappings:
            hook = Template(hook).substitute(self.mappings)
        return hook

    def max_components(self):
        s = set(self.header[HOOK])
        s -= {"LUMA", "ALPHA", "ALPHA_SCALED"}
        if len(s) == 0:
            return 1
        s -= {"CHROMA", "CHROMA_SCALED"}
        if len(s) == 0:
            return 2
        return 4
