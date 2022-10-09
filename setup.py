import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="barl_envs",
    version="0.1.1",
    author="Joshua Evans",
    author_email="jbe25@bath.ac.uk",
    description="A package which provides implementations of various reinforcement learning environments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ueva/BaRL-Envs",
    # packages = setuptools.find_packages(exclude=("example",)),
    packages=["barl_envs"],
    package_dir={"barl_envs": "barl_envs"},
    data_files=[
        (
            "room_files",
            [
                "barl_envs/envs/discrete_rooms/data/two_rooms.txt",
                "barl_envs/envs/discrete_rooms/data/six_rooms.txt",
                "barl_envs/envs/discrete_rooms/data/nine_rooms.txt",
                "barl_envs/envs/discrete_rooms/data/xu_four_rooms.txt",
                "barl_envs/envs/discrete_rooms/data/bridge_room.txt",
                "barl_envs/envs/discrete_rooms/data/cage_room.txt",
                "barl_envs/envs/discrete_rooms/data/empty_room.txt",
                "barl_envs/envs/discrete_rooms/data/small_rooms.txt",
                "barl_envs/envs/discrete_rooms/data/four_rooms.txt",
                "barl_envs/envs/discrete_rooms/data/four_rooms_holes.txt",
                "barl_envs/envs/discrete_rooms/data/maze_rooms.txt",
                "barl_envs/envs/discrete_rooms/data/spiral_room.txt",
                "barl_envs/envs/discrete_rooms/data/parr_maze.txt",
                "barl_envs/envs/discrete_rooms/data/parr_mini_maze.txt",
                "barl_envs/envs/discrete_rooms/data/four_rooms_transfer.txt",
                "barl_envs/envs/discrete_rooms/data/xu_four_rooms_transfer.txt",
                "barl_envs/envs/discrete_rooms/data/nine_rooms_transfer.txt",
            ],
        ),
        (
            "pacman_files",
            ["barl_envs/envs/grid_pacman/data/four_room.txt"],
        ),
    ],
    install_requires=["importlib_resources", "importlib_metadata", "numpy", "networkx", "pygame"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: English",
    ],
)