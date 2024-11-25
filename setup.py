import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="simpleenvs",
    version="0.5.0",
    author="Joshua Evans",
    author_email="jbe25@bath.ac.uk",
    description="A package providing implementations of sequential decision problems using the SimpleOptions framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ueva/BaRL-Envs",
    packages=setuptools.find_packages(exclude=("test")),
    package_dir={"simpleenvs": "simpleenvs"},
    package_data={
        "simpleenvs": [
            "envs/discrete_rooms/data/two_rooms.txt",
            "envs/discrete_rooms/data/six_rooms.txt",
            "envs/discrete_rooms/data/nine_rooms.txt",
            "envs/discrete_rooms/data/xu_four_rooms.txt",
            "envs/discrete_rooms/data/bridge_room.txt",
            "envs/discrete_rooms/data/cage_room.txt",
            "envs/discrete_rooms/data/empty_room.txt",
            "envs/discrete_rooms/data/small_rooms.txt",
            "envs/discrete_rooms/data/four_rooms.txt",
            "envs/discrete_rooms/data/four_rooms_holes.txt",
            "envs/discrete_rooms/data/maze_rooms.txt",
            "envs/discrete_rooms/data/spiral_room.txt",
            "envs/discrete_rooms/data/parr_maze.txt",
            "envs/discrete_rooms/data/parr_mini_maze.txt",
            "envs/discrete_rooms/data/ramesh_maze.txt",
            "envs/discrete_rooms/data/snake_room.txt",
            "envs/continuous_rooms/data/xu_four_rooms.txt",
            "envs/continuous_rooms/data/empty_room.txt",
            "envs/continuous_rooms/data/snake_room.txt",
            "renderers/taxi_renderer_resources/taxi_full.png",
            "renderers/taxi_renderer_resources/taxi_empty.png",
            "renderers/taxi_renderer_resources/passenger.png",
            "renderers/taxi_renderer_resources/goal_flag.png",
            "envs/grid_pacman/data/four_room.txt",
        ]
    },
    include_package_data=True,
    install_requires=[
        "simpleoptions",
        "importlib_resources",
        "importlib_metadata",
        "numpy",
        "networkx",
        "pygame",
        "gymnasium",
        "distinctipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    options={
        "ruff": {
            "line-length": 120,
        }
    },
)
