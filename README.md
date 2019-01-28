# Political-News-Analysis

In this project, we analysis the political news between 2018-07-01 and 2018-12-31 from various media.

The code is written in [Python3](https://www.python.org) with [jupyter notebook](https://jupyter.org/).

## Installation
Download the repository
```bash
$ git clone https://github.com/MiccWan/Political-News-Analysis.git
```
## Packages
- In crawler, we use **_requests_** and **_BeaufifulSoup4_**.
- In text mining, we use **_pandas_**, **_jieba_**, **_sklearn_** and **_mlxtend_**.
- In visualiztion, we use **_networkx_** and **_plotly_**.

Since **__plotly__** is used in our project, you need to set your credentials to use the package:
- In bash:
```bash
$ python
```
 - In python:
```python
import plotly 
plotly.tools.set_credentials_file(username='<YOUR_ACCOUNT>', api_key='<YOUR_API_KEY>')
```

## Dataset
The dataset obtained by crawler is available at this [Google Drive Folder](https://drive.google.com/drive/folders/13BGgHTNmkkUvdOI8XgRiwBBpANPiRFmC?usp=sharing).
