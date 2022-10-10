import glob
import os
import setuptools
from pathlib import Path


def get_package_description():
    """Returns a description of this package from the markdown files."""
    _readme = Path("README.md")
    _history = Path("HISTORY.md")
    if _readme.exists() and _history.exists():
        with open(_readme.resolve(), "r", encoding="utf-8", errors="ignore") as f:
            readme = f.read()
    else:
        readme = "README"
    if _history.exists():
        with open(_history.resolve(), "r", encoding="utf-8", errors="ignore") as f:
            history = f.read()
    else:
        history = "No history yet."
    return f"{readme}\n\n{history}"


def get_scripts_from_bin():
    """Get all local scripts from bin so they are included in the package."""
    return glob.glob("bin/*")


def get_requirements():
    """Returns all requirements for this package."""
    with open("requirements.txt", "r", encoding="utf-8") as f:
        requirements = f.readlines()
    return list(requirements)


def scour_for_file(file_name: str):
    """
    scour_for_file - search every possible location for a file name. Load each line from that filename into a list.
    """
    contents = []
    for root, dirs, files in os.walk("."):
        if file_name in files:
            with open(os.path.join(root, file_name), "r") as f:
                contents = f.readlines()
    assert len(contents) > 0, f"Could not find {file_name} in any of the locations"
    file_contents = [l for l in contents if l.strip()]
    return file_contents


try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError as e:
    print(f"could not read README.md: {e}")
    long_description = get_package_description()


setuptools.setup(
    name="vid2cleantxt",
    author="Peter Szemraj, Jonathan Lehner",
    author_email="szemraj.dev@gmail.com",
    description="A command-line tool to easily transcribe speech-based video files into clean text. also in Colab.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pszemraj/vid2cleantxt",
    include_package_data=True,
    include_dirs=["bin"],
    # package_dir={"": "confectionary"},
    data_files=[("", ["LICENSE"]), ("", ["requirements.txt"]), ("", ["README.md"])],
    packages=setuptools.find_packages(),
    install_requires=scour_for_file("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Topic :: Text Processing",
    ],
    scripts=get_scripts_from_bin(),
    python_requires=">=3.7",
    setuptools_git_versioning={
        "enabled": True,
    },
    setup_requires=["setuptools-git-versioning"],
)
