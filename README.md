# Introduction to Symbolic Music Alignment

This repository contains an very short introduction to symbolic music alignment using dynamic time warping.

The tutorial is based on the lecture on music alignment for the [Musical Informatics](https://www.jku.at/en/institut-fuer-computational-perception/lehre/alle-lehrveranstaltungen/special-topics-musical-informatics/) course offered at JKU (see [repository for the Winter semester 2023](https://github.com/MusicalInformatics/miws23)).

## Setup

The code can be setup locally using the provided conda environment:

```bash
# Clone the repository from GitHub
git clone https://github.com/neosatrapahereje/music_alignment_tutorial.git

cd music_alignment_tutorial

# Create conda environment
conda env create -f environment.yml
```

## Running the tutorial

### Colab

You can run the tutorial directly on Google Colab, without the need to setup the code:

* Symbolic Music Alignment: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neosatrapahereje/music_alignment_tutorial/blob/main/Symbolic_Music_Alignment.ipynb)

* DTW tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neosatrapahereje/music_alignment_tutorial/blob/main/DTW_tutorial.ipynb)

### Locally

You can run the tutorial locally using the following code

```bash
# Go to the directory for the tutorial
cd path/to/music_alignment_tutorial

# Activate environment
conda activate music_alignment_tutorial

# Run the tutorial on symbolic music alignment
jupyter notebook Symbolic_Music_Alignment.ipynb

# Run the tutorial on dynamic time warping
jupyter notebook DTW_tutorial.ipynb
```

## License

The code is provided under the MIT license (see [LICENSE](https://github.com/neosatrapahereje/music_alignment_tutorial/blob/main/LICENSE)).
