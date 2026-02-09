from setuptools import setup

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(
    name='hkg',
    version='0.1',
    author='Gian Carlo Milanese',
    author_email='giancarlomilanese@unimib.com',
    description=("Health Knowledge Graph"),
    url='https://github.com/GianCarloMilanese/health_knowledge_graph',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['hkg'],
    install_requires=required,
    python_requires=">=3.9",
)
