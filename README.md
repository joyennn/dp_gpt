# How to use `dp_gpt.py`

## Initial setting
Clone the repository on Google Colab.
```python
!git clone https://github.com/joyennn/dp_gpt.git
```

Install required libraries.
```python
%cd dp_gpt
!pip install -r requirements.txt --quiet
```

Import the functions from ```dp_gpt```.<br>
There are 5 functions to use : ```size```, ```dp```, ```preview```, ```chat```, ```apply```
```python
from dp_gpt import size, dp, preview, chat, apply
```

The raw data file to be filtered must be a ```.txt``` file where each line contains an English sentence.<br>
Additional files for analysis must also be ```.txt``` files, with words, phrases, or sentences separated by a line.<br>
All required files should be uploaded in the ```dp_gpt``` folder.

<br>

## Functions

### (1) Checking file size: ```size()```
If the raw data file you uploaded is named ```example.txt```, you can check the total number of sentences in the file.
```python
size("example.txt")
```
<br>

### (2) Dependency parsing: ```dp()```
The raw data is dependency-parsed from stanza dependency parsing and assigned in the ```df``` variable as a DataFrame.<br>
When only the filename is provided as an argument, the parser will process up to 100,000 sentences by default.<br>
If the runtime type in Colab is set to GPU, the processing speed increases. If not set, it defaults to CPU processing.
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

<br>

### (3) Preview the dependency parsed dataframe: ```preview()```
After ```dp()```, you can input a variable name and a sentence number to retrieve the parsed result of that specific sentence.<br>
The result will be displayed in DataFrame format.
```python
preview(df, 10)
```

<br>

### (4) Code generation with GPT API: ```chat()```
Using the GPT API, the system generates code to filter a DataFrame based on natural language query.
When a chat session begins, the user inputs commands specifying conditions for inclusion and exclusion separately.
The generated code is assigned to ```inclusion``` & ```exclusion``` variables, respectively.
```python
inclusion, exclusion = chat("your_gpt_api_key")
```
It is also possible to create multiple conditions.
```python
inclusion1, exclusion1 = chat("your_gpt_api_key")
inclusion2, exclusion2 = chat("your_gpt_api_key")
```

<br>

***
### *** The structure of ```chat()``` ***<br>
#### Inclusion
- Input conditions for filtering sentences for inclusion. <br>
- Multiple inclusion conditions can be provided. 
- To finish this step and proceed to the next stage, press "Enter" on an input space.
#### Exclusion
- Input conditions for filtering sentences for exclusion. <br>
- Multiple exclusion conditions can be provided. 
- To finish this step and proceed to the next stage, press 'Enter' on an input space.
#### Generating code
- Based on the provided conditions, the system generates and displays the code.
- If you respond with "yes" or "y"  to the question asking whether to save the generated code, the code is stored in the respective variables.
- If you responds with "no" or "n" to this question, the chat() process restarts from the beginning without saving any previous records.
#### Note
- "Exit" will terminate the chat at any steps.
- "Reset" will reset the chat to its initial state at any steps.
***

## Guide to customizing the code
Sometimes, directly modifying the generated code is simpler than regenerating it from scratch.<br>
In such cases, you can first save the generated code in lists (```inclusion``` & ```exclusion```) and modify it as follows.
### Remove
To remove a code from a list, use ```del``` followed by the list name and the index. (Note that indexing starts from 0.)<br>
For example, to remove the 3rd code from the ```inclusion``` list, use the following command:
```python
del list[index]

del inclusion[2]  #example
```
### Add
To add a new code (value) to a list, use ```append```. (The code should be enclosed in quotes.)<br>
For example, to add a value like ```df['deprel'] == 'root'``` to ```inclusion``` list, use the following command:
```python
list.append(value)

inclusion.append("df['deprel'] == 'root'")  #example
```
### Replace
To replace an existing code with a new one or modify part of the code, you can update the value at a specific index.<br>
For example, if the 3rd code in ```inclusion``` contains a period puctuation (e.g., ```df['deprel'] == 'root'.```), it may be better to remove the period rather than generating a new code.
```python
list[index] = new_value

inclusion[2] = "df['deprel'] == 'root'"
```

<br>

### (5) Apply generated code to the dependency parsed dataframe: ```apply()```
Generated codes ```inclusion``` & ```exclusion``` are applied to the dependency parsed dataframe ```df```.<br>
Review the final filtered sentences and enter a filename (.txt | .csv) to save them.
```python
apply(df, inclusion, exclusion)
```
Multiple ```df``` and ```inclusion``` & ```exclusion``` variables are used as follows.
```python
apply(df1, inclusion, exclusion)
apply(df2, inclusion, exclusion)
apply(df3, inclusion, exclusion)
apply(df4, inclusion, exclusion)
```
```python
apply(df, inclusion1, exclusion1)
apply(df, inclusion2, exclusion2)
```
```python
apply(df1, inclusion1, exclusion1)
apply(df2, inclusion2, exclusion2)
```
