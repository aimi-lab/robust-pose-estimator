source env/bin/activate

python -m pip install vtk==9.0.1
python -m pip install PyQt5
python -m pip install numpy

git clone https://github.com/enthought/mayavi.git
cd mayavi
python setup.py install