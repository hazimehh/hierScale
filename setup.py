import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='hierScale',
     version='1.0.6',
     author="Hussein Hazimeh and Rahul Mazumder",
     author_email="hazimeh@mit.edu",
     description="A scalable package for fitting sparse linear regression models with pairwise feature interactions, under strong hierarchy.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/hazimehh/hierScale",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     python_requires='>=3.0',
     install_requires=["numpy", "scipy", "numba", "networkx"] # "gurobipy" is not available on pip.
 )
