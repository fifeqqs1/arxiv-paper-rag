from setuptools import setup, find_packages

# 读取 requirements.txt
def parse_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="mortyclaw",
    version="1.0.0",
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    entry_points={
        "console_scripts": [
            "mortyclaw=entry.cli:main",
        ],
    },
)
