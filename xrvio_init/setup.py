from setuptools import setup, find_packages
setup(
  name="xrvio_init",
  version="0.1",
  packages=find_packages(),
  install_requires=["numpy","opencv-python","scipy","transforms3d",
                    ] #g2o already installed
)
