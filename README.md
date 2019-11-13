
# lspsuite (Scott's LSP Suite)

**lspsuite** is a set of Python tools (and bash scripts) related to LSP simulations. They may be useful for certain stages of creating and analyzing the simulations.

This suite incorporates code written both by Scott Feister and by Gregory Ngirmang -- the latter, as part of his "lspreader" repository: https://github.com/noobermin/lspreader

The primary use case has been personal, but I'm sharing these tools in case others find them useful.

## Setup

### Dependencies
This module requires **Python 3.6+** and a working implementation of **MPI** (e.g. OpenMPI or MPICH). Installation requires **git**.

**OS X users:** Prior to installing dependencies, ensure an adequate Python installation by following [this guide](https://matplotlib.org/faq/installing_faq.html#osx-notes). The Python that ships with OS X may not work well with some required dependencies.

* [`numpy`](http://www.numpy.org/)
* [`scipy`](https://www.scipy.org/)
* [`matplotlib`](https://matplotlib.org/)
* [`h5py`](https://www.h5py.org/)
* [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/)

The dependencies may be installed according to the directions on 
their webpages, or with any Python
package manager that supports them. For example, one could use `pip` to install
them as
 ```bash
pip install numpy scipy matplotlib h5py mpi4py
```

**NOTE**: If you are on a cluster where you do not have write permissions to the python installation directory, you may need to add "--user" to your pip and setup calls here and below. E.g.
```bash
pip install --user numpy scipy matplotlib h5py mpi4py
```

As an alternate to pip, one could also use [Anaconda Python](https://anaconda.org/anaconda/python) to
install the dependencies
```bash
conda install numpy scipy matplotlib h5py
conda install -c conda-forge mpi4py
```

### Installation
After installing the required packages, we may install **lspsuite**.

One way to install **lspsuite** is via
```bash
pip install git+https://github.com/phyzicist/lspsuite.git
```

To update **lspsuite** at a later date
```bash
pip install --upgrade git+https://github.com/phyzicist/lspsuite.git
```

An alternative way to install **lspsuite** is via
```bash
git clone https://github.com/phyzicist/lspsuite.git
cd lspsuite/
python setup.py install
```

If you installed with the "--user" flag and "\~/.local/bin" is not on your system path, add it to your path (e.g. by appending the line "PATH=$PATH:$HOME/.local/bin" to "\~/.bashrc"). This will allow you to use any command-line tools from this repository (if any exist yet!).

## Usage
You will need to read through the source code for descriptions of the functions of this package. There is unfortunately no formal documentation.

### General usage
```python
import lspsuite as ls

filenames = ls.listp4(".", prefix="flds")
timelist = ls.times(filenames)

domains1, header1 = ls.read_flds("flds41.p4")
domains2, header2 = ls.read_sclr("sclr41.p4")
frames = ls.read_pmovie("pmovie5.p4")
pext = ls.read_pext("pext1.p4")
values, labels = ls.read_history("history.p4")
griddict = ls.read_grid("grid.p4")
regdict = ls.read_regions("regions.p4")
voldict = ls.read_volumes("volumes.p4")
```

### Trajectories from pmovies
First, make a file like this one ("mytraj.py")
```python
import lspsuite as ls

p4dir = "/simulations/run1" # A folder containing pmovie LSP outputs with the initial positions SAK flag on
ls.mpitraj(p4dir) # Creates a file called 'traj.h5' in p4dir
```

Second, call the script with MPI and at least two processors.
```bash
mpirun -np 4 python mytraj.py
```

## Uninstalling

To uninstall **lspsuite**
```shell
pip uninstall lspsuite
```
