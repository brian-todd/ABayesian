import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ABayesian-Brian-Todd",
    version="1.0.0",
    author="Brian Todd",
    description="Bayesian A/B testing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brian-todd/ABayesian",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
