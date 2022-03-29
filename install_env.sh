python3 -m venv env
source env/bin/activate

python3 -m pip install -r requirements.txt
python3 -m pip install -r torchimize/requirements.txt

git clone https://github.com/enthought/mayavi.git
cd mayavi
python3 setup.py install