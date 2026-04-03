# PINNs

## Authors
- <a href="https://scholar.google.com/citations?user=azfl46UAAAAJ&hl=en&oi=en/" target="_blank"> Minh Hieu Do </a> (LPEC)

### Dependencies
```
# info for torch : https://pytorch.org/get-started/locally/
# expl:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# other dep:
uv pip install numpy matplotlib torch-summary scipy seaborn plotly pyvista prompt_toolkit tqdm
```

### Experiments:
 The  benchmarks (or experiments) folder has the structure of a python module. Therefore, to run the test case, use the following command

 - The 5-region fixed source problem

```bash
python -m  benchmarks.one_group.exp_LWR_5S

```

- The simplified C5G7 test case

```bash
python -m  benchmarks.multi_group.exp_C5G7_2D

```
-The TWIGL 2D test case
```bash
python -m  benchmarks.multi_group.exp_TWIGL_2D

```
We can  also go directly to the folder benchmarks (or experiments) and run the test case as:
```bash
python exp_TWIGL_2D.py

```

## Resources

| Path | Description
| :--- | :----------
| [PINNs]() | Main folder.
| &boxvr;&nbsp; [data]() | Data folder.
| &boxvr;&nbsp; [data_analysis]() | Perform some analysis of the results.
| &boxvr;&nbsp; [datagenerators]() | Generate reference solution.
| &boxvr;&nbsp; [notebooks]() | Example notebook for the test cases.
| &boxvr;&nbsp; [benchmarks]()| Some benchmark test cases.
| &boxvr;&nbsp; [experiments]()| Some experiments 
| &boxvr;&nbsp; [result]() | Store the results of the tranning.
| &boxvr;&nbsp; [doc]() | Some documents
| &boxvr;&nbsp; [pyPINNs]()| module PINNs
| &boxv;&nbsp; &boxvr;&nbsp; [Data]() | sampling methods
| &boxv;&nbsp; &boxvr;&nbsp; [Domain]() | Domain calculation
| &boxv;&nbsp; &boxvr;&nbsp; [Mesh]()| Structure of mesh 
| &boxv;&nbsp; &boxvr;&nbsp; [Model]()| Neural Networks
| &boxv;&nbsp; &boxvr;&nbsp; [PDE]()| Primal and Mixed form 
| &boxv;&nbsp; &boxvr;&nbsp; [Tools]()| some useful functions
| &boxv;&nbsp; &boxvr;&nbsp; [Train]()| some train methods
| &boxv;&nbsp; &boxvr;&nbsp; [Xs]()| cross sections
