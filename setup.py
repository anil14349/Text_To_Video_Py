from setuptools import setup, find_packages

setup(
    name="t2v",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "streamlit>=1.24.0",
        "python-docx>=0.8.11",
        "PyPDF2>=3.0.1",
        "docx2txt>=0.8",
        "soundfile>=0.12.1",
        "ChatTTS>=0.2.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "nltk>=3.8.0",
        "spacy>=3.5.0",
        "plotly>=5.3.0",
        "python-dotenv>=0.19.0",
        "rouge-score>=0.1.2",
        "psutil>=5.9.0"
    ],
    python_requires=">=3.7",
)
