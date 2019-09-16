from os.path import dirname, join

from setuptools import find_packages, setup


def read_file(file):
    with open(file, "rt") as f:
        return f.read()


with open(join(dirname(__file__), 'transformer_pytorch/VERSION.txt'), 'rb') as f:
    version = f.read().decode('ascii').strip()
    

setup(
    name='transformer_pytorch',
    version=version,
    description='transformer in pytorch',
    packages=find_packages(exclude=[]),
    author='allen',
    author_email='yujiangallen@126.com',
    license='Apache License v2',
    package_data={'': ['*.*']},
    url='https://github.com/walkacross/transformer-pytorch',
    install_requires=read_file("requirements.txt").strip(),
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
