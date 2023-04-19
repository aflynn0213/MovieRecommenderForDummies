#!/usr/bin/python3
import argparse

#USER CREATED LIBRARIES
from engine import Engine


from surprise.model_selection import train_test_split

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a',"--algorithm",type=int,default=1,help="Algorithm to run with options:{1)gradient, 2)cosine}")
    args = parser.parse_args()
    
    eng = Engine(args.algorithm)
    run = True
    while(run):
        run = eng.run()
    