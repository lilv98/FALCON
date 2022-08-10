#!/bin/bash

for n_inconsistent in 0 1 2 5 10
do  
    python ../model/pizza.py --n_inconsistent $n_inconsistent --n_models 20
done

