for i in {1}
do
    python experiment.py --seed 1 &
    python experiment.py --seed 2 &
    python experiment.py --seed 3 &
    python experiment.py --seed 4 &
    python experiment.py --seed 5 &
    python experiment.py --seed 6 &
    python experiment.py --seed 7 &
    python experiment.py --seed 8 &
    python experiment.py --seed 9 &
    python experiment.py --seed 10 
done