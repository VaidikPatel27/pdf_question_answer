import os
import re
import emoji
import PyPDF2
from nltk.tokenize import sent_tokenize
from semantic_text_splitter import TextSplitter
import nltk
nltk.download('punkt')


def save_data(path, file, type = 'wb'):
	if os.path.exists(path):
		os.remove(path)
		with open(path, type) as f:
			f.write(file)
	else:
		with open(path, type) as f:
			f.write(file)
def pdf_loader(file):
  reader = PyPDF2.PdfReader(file)
  num_pages = len(reader.pages)
  text = ""
  for page_num in range(num_pages):
      page = reader.pages[page_num]
      text += page.extract_text()
  return text
def nlp_preprocessing(txt):

  # Remove new line tag
  txt = txt.replace('\n', '')

  # Lower case and remove extra spaces from the text
  txt = txt.lower().strip()

  for w in txt:
    if emoji.is_emoji(w):
      txt = txt.replace(w, '')

  # Remove HTML tags
  def remove_HTML_tags(txt):
    pattern = re.compile(r'<.*?>')
    return pattern.sub(r'', txt)

  txt = remove_HTML_tags(txt)

  txt = txt.replace("•", '.')

  # remove specific characters
  punctuations = r'''!"#$%&'*+;<=>[\]^_`{|}~'''
  for char in punctuations:
    txt = txt.replace(char,'')

  # tokenizer
  txt = sent_tokenize(txt)

  txt_ls = []
  for sentence in txt:
    txt_ls.append(sentence.strip())

  return ' '.join(txt_ls)
def data_chunks(text):
    len_text = len(text.split())
    print(len_text)
    max_tokens = 6000
    splitter = TextSplitter(trim_chunks=False)
    data_chunks = splitter.chunks(text, max_tokens)
    return data_chunks





