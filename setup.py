from setuptools import setup, find_packages

setup(
    name='YourFlaskApp',
    version='1.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'flask',
        'requests',
        'torch',
        'transformers',
        'torchvision',
        'numpy',
        'Pillow',
        'openai',
    ],
)
