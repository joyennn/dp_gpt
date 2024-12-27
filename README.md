# STANZA_GPT
## How to Use `dp.py`

Clone the repository on Google Colab:
```python
!git clone https://github.com/joyennn/stanza_gpt.git
```

Install required library:
```python
%cd stanza_gpt
!pip install -r requirements.txt
```

Import and use the functions in your Python script:
```python
from dp import process, preview, chat
```

Upload an 'example.txt' file containing English sentences to the stanza_gpt folder.\n
Specify the file path and the number of sentences to parse, and run the code.\n
If the number of sentences to parse is not specified, it defaults to 100,000.\n
```python
process("example.txt", 50000)
or
process("example.txt")
```

Input a sentence ID in 'preview' function to view the dataframe of the sentence's dependency parsing results.
```python
preview(10)
```

Enter the GPT API key you received to interact with the dependency-parsed data in natural language.
```python
chat("your_gpt_api_key")
```
