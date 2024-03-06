cd subsampling
python setup.py build_ext --inplace
cd ..

cd pointops/
python setup.py install
cd ..