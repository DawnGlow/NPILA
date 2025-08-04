# NPILA
Non-profiled Incremental Learning-based Side-channel Attack on Lattice-based KEMs

## Preparation
1. Place the trace file `wave.txt` in the `./data/wave/` directory.  
   Each line in `wave.txt` contains traces for 256 secret key coefficients.

2. Place the label file `guessM.txt` in the `./data/label/` directory.  
   `guessM.txt` stores the message bits corresponding to secret key coefficients.

## Running
Run the following scripts:

```bash
python chunk1.py
python chunk2.py
python fast.py
```

`chunk1.py` corresponds to a chunk size of 1, recovering just one key coefficients at each step.
`chunk2.py` corresponds to a chunk size of 2, recovering two key coefficients at each step.
`fast.py` is a faster version with a chunk size of 2. When the model demonstrates sufficient classification ability, training is stopped early to improve efficiency.
