from setuptools import setup


with open("version.txt", mode="r", encoding="utf-8") as f:
    version = f.read().strip()



setup(
    name="cntell",
    version=version,
    description="See and Tell: AI-driven Assistant to Experience Visual Content",
    author="Aleksandr Lobanov, Ayhem Bouabid",
    author_email="dev@alobanov.space, bouabidayhem@gmail.com",
    entry_points={
         'console_scripts': [
            'cntell=cntell.cntell:main'
        ]
    },
    package_dir={"cntell": "src"},
    package_data={
        "embeddings": ["embeddings"],
        "": ["Dockerfile", "LICENSE", "README.md", "requirements.txt", "version.txt"]
    },
    dependency_links=[
        "https://download.pytorch.org/whl/cu117"
    ],
    install_requires=[
        "torch",
        "torchvision", 
        "torchaudio",
        "numpy",
        "stanza",
        "nltk",
        "datasets",
        "transformers",
        "soundfile",
        "ffmpegio",
        "ffmpeg-python",
        "facenet-pytorch",
        "moviepy",
        "pyannote.audio @ https://github.com/pyannote/pyannote-audio/archive/refs/heads/develop.zip",
    ],
    url="https://github.com/teexone/see-and-tell"
)