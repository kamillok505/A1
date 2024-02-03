import sys
import numpy as np
from utils import data_slicer
from back_propagation import create_neural_net, back_propagation
from MLR import mlr

def main(args):
    print("Arguments:", args)
    if len(args) < 1:
        print("Error: the program expects 1 argument.")
        print("Usage: python run_bp.py parameters_file_path")
        return
    elif len(args) > 1:
        print("Warning: too many arguments (1 expected). Only the first one will be considered.")

    path = args[0]
    data_path = ""
    boundary = 0.8
    folds = 4
    layers = []
    epochs = 0
    η = 0.1
    α = 0.9

    with open(path, 'r') as file:
        for line_num, line in enumerate(file, start=1):
            if line_num == 1:
                data_path = line.strip()
            elif line_num == 2:
                chunks = line.split()
                boundary = float(chunks[0])
            elif line_num == 3:
                folds = int(line.strip())
            elif line_num == 4:
                layers = [int(x) for x in line.split()]
            elif line_num == 5:
                epochs = int(line.strip())
            elif line_num == 6:
                chunks = line.split()
                η = float(chunks[0])
                α = float(chunks[1])

    preprocess = False
    if "A1-turbine.txt" not in data_path and "A1-synthetic.txt" not in data_path:
        preprocess = True

    data = data_slicer(data_path, boundary, preprocess)
    layers.insert(0, data.train.shape[1] - 1)
    nn = create_neural_net(layers)

if __name__ == "__main__":
    main(sys.argv[1:])
