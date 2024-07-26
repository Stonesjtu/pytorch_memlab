from setuptools import setup, find_packages

try:
    long_description = open('README.md').read()
except FileNotFoundError:
    long_description = ''

setup(
    name='pytorch-memlab',
    version='0.3.0',
    licence='MIT',
    description='A lab to do simple and accurate memory experiments on pytorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        ],
    keywords='pytorch memory profile',
    author='Kaiyu Shi',
    author_email='skyisno.1@gmail.com',
    url='https://github.com/Stonesjtu/pytorch_memlab',
    license='MIT',
    include_package_data=True,
    zip_safe=True,
    python_requires='>=3.8',
    install_requires=[
        'setuptools',
        'calmsize',
        'pandas',
        'torch>=2.0',
    ],
    extras_require={
        'ipython': ['IPython>=0.13'],
        'test': ['pytest'],
    },
    packages=find_packages(),
)
