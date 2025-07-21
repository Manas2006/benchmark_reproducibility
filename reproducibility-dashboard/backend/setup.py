from setuptools import setup, find_packages

setup(
    name="reproducibility-dashboard",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "jinja2==3.1.2",
        "pydantic==2.5.0",
        "python-multipart==0.0.6",
        "aiofiles==23.2.1",
        "typer==0.9.0",
        "rich==13.7.0",
        "pytest==7.4.3",
        "pytest-asyncio==0.21.1",
    ],
    entry_points={
        "console_scripts": [
            "reproducibility-cli=app.cli:app",
            "repro-cli=app.cli:app",
        ],
    },
    python_requires=">=3.8",
    author="Reproducibility Dashboard Team",
    author_email="team@reproducibility-dashboard.com",
    description="A full-stack dashboard for hyperparameter sweeps and reproducibility",
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/reproducibility-dashboard/reproducibility-dashboard",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
