from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
    name='qsim',
    version='1.0',
    description='Quantum simulation for NISQ computers',
    long_description=long_description,
    author='Leo Zhou, Madelyn Cain',
    author_email='lzhou@g.harvarad.edu, mcain@g.harvard.edu',
    packages=['qsim'],
    install_requires=['numpy>=1.16.4'], #external dependencies
    scripts=[
            'scripts/basic',
           ]
)