import setuptools

with open("README.md", "r") as f :
    long_description = f.read()

setuptools.setup(
    name = "barl_envs",
    version = "0.0.1",
    author = "Joshua Evans",
    author_email = "jbe25@bath.ac.uk",
    description = "A package which provides implementations of various reinforcement learning environments.",
    long_description = long_description,
    long_description_content_type="text/markdown",
    url = "https://github.com/Ueva/BaRL-Envs",
    #packages = setuptools.find_packages(exclude=("example",)),
    packages = ["barl_envs"],
    package_dir = {"barl_envs": "barl_envs"},
    data_files = [("room_files", ["barl_envs/envs/discrete_rooms/data/two_rooms.txt", "barl_envs/envs/discrete_rooms/data/six_rooms.txt"])],
    install_requires = ["numpy"],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: English"
    ]
)