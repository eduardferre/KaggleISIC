import io
import os
import re
from configparser import ConfigParser

from setuptools import find_packages, setup

# Read metadata from setup.cfg
conf = ConfigParser()
conf.read(["setup.cfg"])
metadata = dict(conf.items("metadata"))


def read(filename):
    """Read a file and replace reStructuredText field lists with plain text."""
    filename = os.path.join(os.path.dirname(__file__), filename)
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(r":[a-z]+:`~?(.*?)`", r"``\1``", fd.read())


setup(
    name=metadata.get("name", "kaggleisic"),
    version=metadata.get("version", "0.1.0"),
    url=metadata.get("url", ""),
    license=metadata.get("license", ""),
    author=metadata.get("author", ""),
    author_email=metadata.get("author_email", ""),
    description=metadata.get("description", ""),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src", exclude=("tests",)),
    package_dir={"": "src"},
    install_requires=[],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
