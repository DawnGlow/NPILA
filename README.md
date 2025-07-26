# NPILA
Non-profiled Incremental Learning-based Side-channel Attack on Lattice-based KEMs

# Preparation
Place the trace file wave.txt in the ./data/wave/ directory.
Each line of wave.txt contains the trace corresponding to 256 coefficients of the secret key.
Place the label file guessM.txt in the ./data/label/ directory.
guessM.txt stores the message bits corresponding to the secret key coefficients, generated with newly chosen ciphertexts.

# Running
Run the following scripts to perform the analysis:
python chunk1.py
python chunk2.py
python fast.py

The result will be saved to:
./result/sk_sum.xlsx
