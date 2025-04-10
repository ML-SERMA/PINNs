# PINNs

**

## Authors
- <a href="https://scholar.google.com/citations?user=azfl46UAAAAJ&hl=en&oi=en/" target="_blank"> Minh Hieu Do </a> (LPEC)
- 
-

### Dependencies
Please make sure you have a miniconda environment installed and the following necessary dependencies (available through pip or conda):
- <a href="https://pytorch.org/" target="_blank"> pytorch </a>
- <a href="https://numpy.org/install/" target="_blank"> numpy </a>
- <a href="https://matplotlib.org/stable/index.html" target="_blank"> matplotlib </a>

### Experiments:
 The experiments folder has the structure of a python module. Therefore, to run the experiments, use the following command

- Center test case

```bash
python -m experiments.mixed_form.exp_mixed_FCN_Center.py

```
-Dauge test case
```bash
python -m experiments.mixed_form.exp_mixed_FCN_Dauge.py

```
We can  also go directly to the folder experiments and run the test case as:
```bash
python exp_mixed_FCN_Center.py

```

## Resources

| Path | Description
| :--- | :----------
| [PINNs]() | Main folder.
| &boxvr;&nbsp; [data]() | Data folder.
| &boxvr;&nbsp; [data_analysis]() | Perform some analysis of the results.
| &boxvr;&nbsp; [datagenerators]() | Generate reference solution.
| &boxvr;&nbsp; [notebooks]() | Example notebook for the test cases.
| &boxvr;&nbsp; [result]() | Store the results of the tranning.
| &boxvr;&nbsp; [doc]() | Some documents
| &boxvr;&nbsp; [experiments]()| Some experiments test cases.
| &boxvr;&nbsp; [pyPINNs]()| module PINNs
| &boxv;&nbsp; &boxvr;&nbsp; [Data]() | sampling methods
| &boxv;&nbsp; &boxvr;&nbsp; [Domain]() | Domain calculation
| &boxv;&nbsp; &boxvr;&nbsp; [Mesh]()| Structure of mesh 
| &boxv;&nbsp; &boxvr;&nbsp; [Model]()| Neural Networks
| &boxv;&nbsp; &boxvr;&nbsp; [PDE]()| Primal and Mixed form 
| &boxv;&nbsp; &boxvr;&nbsp; [Tools]()| some useful functions
| &boxv;&nbsp; &boxvr;&nbsp; [Train]()| some train methods
