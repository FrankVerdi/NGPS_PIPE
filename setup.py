import os
from setuptools import setup

# Define version template in case no version control info is available
VERSION_TEMPLATE = '0.0.0'  # Replace with your desired template version string

try:
    from setuptools_scm import get_version
    version = get_version(root='..', relative_to=__file__)
except Exception:
    version = VERSION_TEMPLATE

# Set up the package with version control handling via setuptools_scm
setup(
    name='ngps_pipe',  # Your package name
    version=version,
    packages=['ngps_pipe'],  # Add your package directory here
    use_scm_version={
        'write_to': os.path.join('ngps_pipe', 'version.py'),
        'write_to_template': VERSION_TEMPLATE
    },
    install_requires=[
        # List of dependencies your package needs
    ],
)
