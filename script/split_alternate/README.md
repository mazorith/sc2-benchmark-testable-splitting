This is simply an alternate server-client for the 
purpose of collecting stats on split models/datasets. 
Data is sent in between as a dictionary, in the form 
of `message = dict{'timestamp' + 'data'}`. 

Additionally, add symlinks for Data/ and dev/ to this directory.

Basic Usage:

```python server.py```

```python client.py```

In the offline case, `client.py` will contain the entire
model and test the end-to-end time of the model 
as well as an evaluation of the model's performance. Evaluators
and the offline case itself will be contained in `client.py` 
only to avoid any duplication of code in `server.py`; in other
words, there is no offline evaluation for `server.py`.

Eventually, have TorchScript-ed models. 