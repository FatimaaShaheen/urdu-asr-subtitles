from pathlib import Path

folders = [
    "configs",
    "notebooks",
    "src/data",
    "src/training",
    "src/inference",
    "src/evaluation",
    "backend",
    "frontend",
    "data/raw",
    "data/processed",
    "models",
    "docs",
    "tests",
]

files = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "docs/glossary.md",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/training/__init__.py",
    "src/inference/__init__.py",
    "src/evaluation/__init__.py",
    "backend/__init__.py",
    "backend/main.py",
    "frontend/app.py",
]

for f in folders:
    Path(f).mkdir(parents=True, exist_ok=True)
for f in files:
    Path(f).touch(exist_ok=True)

print("Project structure ready!")