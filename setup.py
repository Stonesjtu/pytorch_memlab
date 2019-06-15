from setuptools import setup

try:
    long_description = open('README.md').read()
except FileNotFoundError:
    long_description = ''

setup(
    name='pytorch_memlab',
    version='0.0.3',
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
    install_requires=[
        'setuptools',
        'calmsize',
    ],
    packages=['pytorch_memlab'],
)
