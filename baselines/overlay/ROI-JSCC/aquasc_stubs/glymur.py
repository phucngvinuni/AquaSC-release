# Minimal stub for the `glymur` package, required ONLY by
# model/JPEG2000.py (the ROIJPEG2000 / JPEG2000 baselines). AquaSC never
# instantiates those classes; it uses ROIJSCC, which does not touch JPEG2000.
#
# Importing model.JSCC executes `from .JPEG2000 import jpeg2000_patchwise_compress`,
# and JPEG2000.py does `import glymur` at module level. This stub lets the
# canonical `from model.JSCC import ROIJSCC` import path succeed without pulling
# the real glymur / libopenjp2 dependency into the siso-project-env environment.
#
# If a JPEG2000 model is ever actually run, this stub raises a loud error.


class Jp2k:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "glymur stub: JPEG2000 (ROIJPEG2000/JPEG2000) is not used by AquaSC. "
            "Install the real glymur + libopenjp2 to use the JPEG2000 baselines."
        )


def __getattr__(name):
    raise AttributeError(
        f"glymur stub does not implement '{name}'; JPEG2000 baselines are disabled."
    )
