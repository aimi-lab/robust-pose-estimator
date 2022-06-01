python3 -m pip install --upgrade pip

python3 -m venv env
source env/bin/activate

git submodule update --init --recursive

python3 -m pip install -r requirements.txt

git clone https://github.com/enthought/mayavi.git
cd mayavi
python3 setup.py install