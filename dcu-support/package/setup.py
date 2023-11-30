from setuptools import setup, find_packages

setup (
    name = "fastllm_pytools",
    version = "0.0.1",
    description = "Fastllm pytools",
    packages = ['fastllm_pytools'],
    url = "https://developer.hpccube.com/codes/aicomponent/fastllm",
    package_data = {
        '': ['*.dll', '*.so']
    }
)
