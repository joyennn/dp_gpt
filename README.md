# STANZA_GPT
## How to use `stanza_gpt.py`

### (1) Initial setting
Clone the repository on Google Colab.
```python
!git clone https://github.com/joyennn/stanza_gpt.git
```

Install required libraries.
```python
%cd stanza_gpt
!pip install -r requirements.txt
```

Import the functions from 'stanza_gpt.'<br>
There are 4 functions to use : ```dp```, ```preview```, ```chat```, ```extract```
```python
from stanza_gpt import dp, preview, chat, extract
```
<br><br>
### (2) Dependency parsing: ```dp()```
Upload an 'example.txt' file containing English sentences to the 'stanza_gpt' folder.<br>
Specify the file path and the number of sentences to parse, and run the code.
```python
df = dp("example.txt", 50000)
```
If the number of sentences to parse is not specified, it defaults to 100,000.<br>
```python
df = dp("example.txt")
```
<br><br>
### (3) Preview the dependency parsed dataframe: ```preview()```
Input ```df``` and ```a sentence ID``` to view the specific dependency parsed dataframe.
```python
preview(df, 10)
```
<br><br>
### (4) Code generation with GPT API: ```chat()```
Enter the GPT API key you received between the quotation marks.<br>
Write prompts in natural language and get codes that GPT generates.<br>
Ensure that you have final lists of conditional codes - ```inclusion``` & ```exclusion```
```python
inclusion, exclusion = chat("your_gpt_api_key")
```
<br><br>
### (5) Apply generated codes to the dependency parsed dataframe: ```extract()```
Apply generated codes ```inclusion``` & ```exclusion``` to the dependency parsed dataframe ```df```<br>
Review the final filtered sentences and enter a filename (.txt | .csv) to save them.
```python
extract(df, inclusion, exclusion)
```
