conda update -y conda
conda install -y pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 dgl==0.9.0 ogb==1.3.3 gurobi==9.5.2  -c pytorch -c dglteam/label/cu113 -c gurobi -c conda-forge
pip install pulp==2.6.0
cd AdaQP/util/quantization/
python setup.py install
cd ../../..