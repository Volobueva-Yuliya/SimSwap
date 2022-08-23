from setuptools import setup, find_packages


setup(
    name='dcsimswap',
    version='0.1dev',
    packages=find_packages(),
    license='',
    python_requires=">=3.8.*",
    install_requires=[
        'torch>=1.7',
        'opencv-python',
        'torchvision',
        'imageio',
        'moviepy',
        'numpy',
        'insightface',
        'pillow',
    ],
    include_package_data=True,
    package_data={'': ['*.yaml']},
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown'
)
