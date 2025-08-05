# CPU
python3 benchmark.py 100 -b 120 -IRE
python3 benchmark.py 1000 -b 300 -IRE
python3 benchmark.py 5000 -b 600
python3 benchmark.py 8500 -b 1200
# GPU
python3 benchmark.py 100 -b 60 -IRE --gpu
python3 benchmark.py 1000 -b 120 -IRE --gpu
python3 benchmark.py 5000 -b 300 --gpu
python3 benchmark.py 8500 -b 600 --gpu
