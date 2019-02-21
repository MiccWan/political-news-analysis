# Political-News-Analysis

In this project, we analysis the political news between 2018-07-01 and 2018-12-31 from various media.

The code is written in [Python3](https://www.python.org) with [jupyter notebook](https://jupyter.org/).

## Installation
Download the repository
```bash
$ git clone https://github.com/MiccWan/Political-News-Analysis.git
```
Since **__plotly__** is used in our project, you need to set your credentials to use the package:
```bash
$ python
```
then in python interactive shell,
```python
>> import plotly 
>> plotly.tools.set_credentials_file(username='<YOUR_ACCOUNT>', api_key='<YOUR_API_KEY>')
```

## Packages
- In crawler, we use **_requests_** and **_BeaufifulSoup4_**.
- In text mining, we use **_pandas_**, **_jieba_**, **_sklearn_** and **_mlxtend_**.
- In visualiztion, we use **_networkx_** and **_plotly_**.

## Dataset
The dataset obtained by crawler is available at this [Google Drive Folder](https://drive.google.com/drive/folders/13BGgHTNmkkUvdOI8XgRiwBBpANPiRFmC?usp=sharing).

## Folders
- **/crawler**: Crawlers for new_talk and liberty_times.
- **/data**: Storing the list of events, people and reputation data for future analysis.
- **/final_demo**: Report and data for final demo.
- **/jieba_data**: Some dictionary for term frequency analysis.
- **/politicians**: Images and wordcloud of politicians.
- **/tools**: Some tools, modules and test file created in analysing.
