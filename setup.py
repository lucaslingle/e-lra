from io import open

from setuptools import find_packages
from setuptools import setup


setup(
    name="long-range-arena",
    version="3.0.0",
    url="https://github.com/lucaslingle/e-lra/",
    author="Lucas Dax Lingle",
    author_email="lucasdaxlingle@gmail.com",
    description="Working fork of Long Range Arena.",
    long_description=open("README.md", encoding="utf-8").read(),
    packages=find_packages(where="."),
    package_dir={"": "."},
    platforms="any",
    python_requires=">=3.8",
    install_requires=[
        "flax>=0.2.8,<=0.3.6",
        "ml-collections>=0.1.0",
        "tensorboard>=2.3.0",
        "tensorflow>=2.3.1",
        "tensorflow-datasets==4.8.3",
        "tensorflow-hub>=0.15.0",
        "tensorflow-text>=2.7.3",
        "gin-config==0.5.0",
        "attrs==23.1.0",
    ],
    extras_require={
        "cpu": [
            "jaxlib==0.3.0",
            "jax==0.3.0",
        ],
        "cuda11": [
            "jaxlib==0.3.0+cuda11.cudnn82",
            "jax[cuda11_pip]==0.3.0",
        ],
        "tpu": [
            "libtpu-nightly==0.1.dev20220128",
            "jaxlib==0.3.0",
            "jax[tpu]==0.3.0",
            "protobuf<=3.20.1",
        ],
        "dev": [
            "pre-commit",
        ],
    },
)
