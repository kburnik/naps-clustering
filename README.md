## NAPS & NAPS BE Stable Clustering via Monte-Carlo Simulation

This is a snapshot of the NAPS & NAPS BE stable clustering solution using the
Monte-Carlo method.

The datasets are not included in this solution and should be obtained separately
from the Nencki Institute (http://en.nencki.gov.pl/).

* Author: [Kristijan Burnik](http://bit.ly/kristijan-burnik-linkedin-zr)
* Mentor: [Doc. dr. sc. Marko horvat, v. pred.](http://bit.ly/marko-horvat-zr)

## About the project

The project was created as a python library (module).

The main program (entry point) is clustering/analysis.py which contains the
program snippets used to generate tables and graphs from the final
work, while the remaining files are parts of the library.

For further research, it is recommended to start by adding a snippet to
clustering/analysis.py, and if necessary make a separate project or script that
imports existing modules.

## The results

The results are saved in the clustering/out directory. Most of the results are
generated with the cache mechanism to make it easier to modify plots and the
rest of the code that does not affect results that are precomputed for a long
time. Cache files are recognizable by the suffix `.cached-result.json`.

The clustering results are saved in CSV files with one blank line in between
two partitions. It is recommended to use partition indexes for software
solutions from the corresponding cached-result.json file. The order there is the
same as the input the order of the corresponding input data, and the values ​​
denote the partition indices.

## Installation

Windows:

* [Python 3.5.x](https://www.python.org/downloads/release/python-353/#files)
* [Git SCM (git bash)](https://git-scm.com/download/win)
* [MSVC++ 2015 Redistributable](https://www.microsoft.com/en-us/download/details.aspx?id=53587)
* [SciPy](http://stackoverflow.com/a/32064281/2041634)
* [NumPy+MKL](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy) - uninstall numpy first.

** After installing the above, everything else needs to be done in git-bash. **

Linux:

* The steps are almost the same, you need to install the software with apt-get
  and pip while MSVC++ 2015 Redistributable is not required.

## Fetching the code and preparing the development environment

```bash
cd ~
git clone https://github.com/kburnik/naps-clustering

cd ~/naps-clustering
virtualenv venv -p /c/python3X/python.exe  # Adjust path python3.X
```

## Initializing the environment (git bash & python virtualenv)

This needs to be done only once at the beginning of a terminal session.

```bash
cd ~/naps-clustering
. venv/Scripts/activate # . venv/bin/activate on Linux-u
```

## Running

```sh
# The program will print the available snippets to run
python -u clustering/analysis.py

# Run the first snippet.
python -u clustering/analysis.py 1
```
