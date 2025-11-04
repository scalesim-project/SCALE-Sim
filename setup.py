import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='scalesim',
    version='3.0.0',
    description='Systolic CNN AcceLerator Simulator',
    long_description=README,
    long_description_content_type="text/markdown",
    author='ritikraj7, anands09, tushar',
    author_email='ritik.raj@gatech.edu',
    maintainer='SynergyLab, GT',
    maintainer_email='ritik.raj@gatech.edu',
    url='https://github.com/scalesim-project/SCALE-Sim',
    license="MIT",
    packages=find_packages(),
    include_package_data=False,                                 # The include_package_data argument controls whether non-code files are copied when your package is installed
    install_requires=["numpy","configparser","absl-py", "tqdm", "pandas", "setuptools", "matplotlib", "cython", "numba"],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering'
    ]
)
