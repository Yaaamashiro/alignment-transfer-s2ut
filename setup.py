"""Bundle the canonical research config without a second editable default."""
from pathlib import Path
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        source = Path(__file__).parent / "configs/experiments/conditions.yaml"
        target = Path(self.build_lib) / "alignment_transfer_s2ut/conditions.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


setup(cmdclass={"build_py": BuildPy})
