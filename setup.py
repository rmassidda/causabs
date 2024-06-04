from setuptools import setup, find_packages

setup(
    name='causabs',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.2',
        'igraph==0.11.3',
    ],
    author='Riccardo Massidda',
    author_email='riccardo.massidda@phd.unipi.it',
    description='Linear Causal Abstraction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/rmassidda/causabs',  # Update with your package's URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)

