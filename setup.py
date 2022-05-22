from setuptools import find_packages, setup

setup(
    name = 'mle_training',
    version = 0.2,
    packages = find_packages(where="src"),
    package_dir = {"":"src"}
)