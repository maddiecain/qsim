from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='qsim',
    version='1.0',
    description='Quantum simulation for NISQ computers',
    long_description=long_description,
    author='Madelyn Cain',
    author_email='mcain@g.harvard.edu',
    packages=find_packages('.')
)