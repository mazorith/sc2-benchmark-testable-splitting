This is simply an alternate server-client for the 
purpose of collecting stats on split models/datasets. 
Data is sent as a dictionary, where it would be `d['data']`
for simple bytes but `d['data_tensor']` for pytorch tensors
since those require different measurements. 

Basic Usage:

```python server.py```

```python client.py```