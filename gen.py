import nnedi3
import superxbr


def run(hooks, suffix):
    gen = superxbr.SuperxBR(hook=hooks)
    with open("superxbr%s.hook" % suffix, "w") as f:
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
                for step in list(nnedi3.Step):
                    f.write(gen.generate(step))


run(["LUMA"], "")
run(["CHROMA"], "-chroma")
run(["LUMA", "CHROMA"], "-yuv")
run(["LUMA", "CHROMA", "RGB", "XYZ"], "-all")
