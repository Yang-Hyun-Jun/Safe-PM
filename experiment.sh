for i in {1}
do
    python experiment.py --seed 1 &
    python experiment.py --seed 2 &
    python experiment.py --seed 3 &
    python experiment.py --seed 4 &
    python experiment.py --seed 5 
done

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

for i in {1}
do  
    python experiment.py --freq 600 --seed 1 &
    python experiment.py --freq 700 --seed 2 &
    python experiment.py --freq 900 --seed 3 &
    python experiment.py --freq 1000 --seed 4 &
    python experiment.py --freq 1200 --seed 5 &
    python experiment.py --freq 1400 --seed 6 &
    python experiment.py --freq 1500 --seed 7 &
    python experiment.py --freq 1600 --seed 8 &
    python experiment.py --freq 1700 --seed 9 &
    python experiment.py --freq 1800 --seed 10 &

done