import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="simpleenvs",
    version="0.2.0",
    author="Joshua Evans",
    author_email="jbe25@bath.ac.uk",
    description="A package which provides implementations of various reinforcement learning environments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ueva/BaRL-Envs",
    # packages=setuptools.find_packages(exclude=("example", "test")),
    packages=["simpleenvs"],
    package_dir={"simpleenvs": "simpleenvs"},
    data_files=[
        (
            "room_files",
            [
                "simpleenvs/envs/discrete_rooms/data/two_rooms.txt",
                "simpleenvs/envs/discrete_rooms/data/six_rooms.txt",
                "simpleenvs/envs/discrete_rooms/data/nine_rooms.txt",
                "simpleenvs/envs/discrete_rooms/data/xu_four_rooms.txt",
                "simpleenvs/envs/discrete_rooms/data/bridge_room.txt",
                "simpleenvs/envs/discrete_rooms/data/cage_room.txt",
                "simpleenvs/envs/discrete_rooms/data/empty_room.txt",
                "simpleenvs/envs/discrete_rooms/data/small_rooms.txt",
                "simpleenvs/envs/discrete_rooms/data/four_rooms.txt",
                "simpleenvs/envs/discrete_rooms/data/four_rooms_holes.txt",
                "simpleenvs/envs/discrete_rooms/data/maze_rooms.txt",
                "simpleenvs/envs/discrete_rooms/data/spiral_room.txt",
                "simpleenvs/envs/discrete_rooms/data/parr_maze.txt",
                "simpleenvs/envs/discrete_rooms/data/parr_mini_maze.txt",
                "simpleenvs/envs/discrete_rooms/data/ramesh_maze.txt",
            ],
        ),
        (
            "pacman_files",
            ["simpleenvs/envs/grid_pacman/data/four_room.txt"],
        ),
    ],
    install_requires=[
        "simpleoptions",
        "importlib_resources",
        "importlib_metadata",
        "numpy",
        "networkx",
        "pygame",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: English",
    ],
    python_requires=">=3.6",
)
