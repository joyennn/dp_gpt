# How to use `dp_gpt.py`

## Initial setting
Clone the repository on Google Colab.
```python
!git clone https://github.com/joyennn/dp_gpt.git
```

Install required libraries.
```python
%cd dp_gpt
!pip install -r requirements.txt
```

Import the functions from ```dp_gpt```.<br>
There are 5 functions to use : ```size```, ```dp```, ```preview```, ```chat```, ```apply```
```python
from dp_gpt import size, dp, preview, chat, apply
```

The raw data file to be filtered must be a ```.txt``` file where each line contains an English sentence.<br>
If additional files are required for analysis, they must also be ```.txt``` files, with words, phrases, or sentences separated by a line.<br>
All required files should be uploaded in the ```dp_gpt``` folder.

<br>

## Functions

### (1) Checking file size: ```size()```
If the raw data file you uploaded is named ```example.txt```, you can check the total number of sentences in the file.
```python
size("example.txt")
```

### (2) Dependency parsing: ```dp()```
The raw data is dependency-parsed and stored in the df variable as a DataFrame.<br>
When only the filename is provided as an argument, the parser will process up to 100,000 sentences by default.
```python
df = dp("example.txt")
```

If a number is specified with the file, the parser will process sentences from the beginning of the file up to the given number.
```python
df = dp("example.txt", 50000)
```

If two numbers are specified with the file, the parser will process sentences from the first number (start) to the second number (end).
```python
df = dp("example.txt", 50001, 60000)
```

If a number and ```None``` are specified with the file, the parser will process sentences from the given number to the end.
```python
df = dp("example.txt", 60001, None)
```

To process more than 100,000 sentences, assign each parsed portion to different variables.
```python
df1 = dp("example.txt", 100000)
df2 = dp("example.txt", 100001, 200000)
df3 = dp("example.txt", 200001, 300000)
df4 = dp("example.txt", 300001, None)
```

### (3) Preview the dependency parsed dataframe: ```preview()```
After ```dp()```, you can input a variable name and a sentence number to retrieve the parsed result of that specific sentence.<br>
The result will be displayed in DataFrame format.
```python
preview(df, 10)
```
### (4) Code generation with GPT API: ```chat()```
Enter the GPT API key you received between the quotation marks.<br>
Write prompts in natural language and get codes that GPT generates.<br>
Ensure that you have final lists of conditional codes - ```inclusion``` & ```exclusion```.
```python
inclusion, exclusion = chat("your_gpt_api_key")
```
<br><br>
### (5) Apply generated codes to the dependency parsed dataframe: ```apply()```
Apply generated codes ```inclusion``` & ```exclusion``` to the dependency parsed dataframe ```df```.<br>
Review the final filtered sentences and enter a filename (.txt | .csv) to save them.
```python
extract(df, inclusion, exclusion)
```
