import re
import setuptools
from pkg_resources import get_distribution, DistributionNotFound

with open("README.md", "r") as fh:
    long_description = fh.read()

INSTALL_REQUIRES = [
    "accelerate==0.28.0",
    "beautifulsoup4==4.12.3",
    "einops==0.7.0",
    "ftfy==6.2.0",
    "ninja==1.11.1.1",
    "protobuf==5.27.1",
    "scipy==1.12.0",
    "torch==2.1.0",
    "torchinfo==1.8.0",
    "torchvision==0.16.0",
    "tqdm==4.66.2",
    "transformers==4.40.2",
    "sentencepiece==0.2.0",
    "piq==0.8.0",
    "soundata==1.0.1",
    "soundfile==0.12.1",
    "librosa==0.10.2.post1",
    "msclap==1.3.3",
]

ALT_INSTALL_REQUIRES = {}


def check_alternative_installation(install_require, alternative_install_requires):
    """If some version version of alternative requirement installed, return alternative,
    else return main.
    """
    for alternative_install_require in alternative_install_requires:
        try:
            alternative_pkg_name = re.split(r"[!<>=]", alternative_install_require)[0]
            get_distribution(alternative_pkg_name)
            return str(alternative_install_require)
        except DistributionNotFound:
            continue

    return str(install_require)


def get_install_requirements(main_requires, alternative_requires):
    """Iterates over all install requires
    If an install require has an alternative option, check if this option is installed
    If that is the case, replace the install require by the alternative to not install dual package
    """
    install_requires = []
    for main_require in main_requires:
        if main_require in alternative_requires:
            main_require = check_alternative_installation(
                main_require, alternative_requires.get(main_require)
            )
        install_requires.append(main_require)

    return install_requires


INSTALL_REQUIRES = get_install_requirements(INSTALL_REQUIRES, ALT_INSTALL_REQUIRES)

setuptools.setup(
    name="xdiffusion",
    version="0.0.1",
    author="Sam Wookey",
    author_email="sam@thinky.ai",
    description="Unified Diffusion Models.",
    long_description_content_type="text/markdown",
    long_description=long_description,
    license_files=("LICENSE",),
    license="Copyright 2024, Sam Wookey",
    url="https://github.com/swookey-thinky/xdiffusion",
    packages=setuptools.find_packages(
        exclude=["config", "docs", "sampling", "tools", "training"]
    ),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.6",
)
