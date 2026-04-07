from setuptools import find_packages, setup

package_name = "ur_rl_sb3"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/ur_rl_sim.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="bophy",
    maintainer_email="you@example.com",
    description="Stable-Baselines3 reinforcement learning examples for UR simulation in Gazebo.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "train_sac = ur_rl_sb3.train_sac:main",
            "eval_sac = ur_rl_sb3.eval_sac:main",
        ],
    },
    options={
        'build_scripts': {
        'executable': '/home/bophy/d2lros2/.rosvenv/bin/python3',
        },
    },
)
