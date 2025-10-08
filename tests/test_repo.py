import pathlib

def test_readme_exists():
    assert pathlib.Path("README.md").exists()

def test_requirements_exists():
    assert pathlib.Path("requirements.txt").exists()

