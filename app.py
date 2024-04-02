import os
import streamlit as st
import numpy as np
from src.data_preprocessing import (
									save_data,
									pdf_loader,
									nlp_preprocessing,
									data_chunks
									)
from src.openai import (
								text_embedding,
								get_response
							  )
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

st.title("Ask question to the pdf!")
st.subheader("Ask here!")
st.sidebar.title("Upload PDF File")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

token = os.getenv("HUGGINGFACE_TOKEN")
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", token=token)

if uploaded_file:
	file_name = uploaded_file.name
	outputs = "/workspaces/pdf_question_answer/data/output"
	path = os.path.join(outputs, f"{file_name}.pdf")
	save_data(path, uploaded_file.getvalue(), type = 'wb')
	data = pdf_loader(path)
	text = nlp_preprocessing(data)
	data_chunks = data_chunks(text)
	encoded_data = text_embedding(data_chunks)
	index = faiss.IndexFlatL2(384)
	index.add(encoded_data.reshape(encoded_data.shape[0], 384))
else:
	st.write("")

query = st.text_input("", placeholder="Write your query here!")


if query:
	def search(query, top_searches = 5):
		query_encode = text_embedding([query])
		k = min(top_searches, len(data_chunks))
		d, idx = index.search(query_encode, k)
		l = []
		for i in idx[0]:
			l.append(data_chunks[i])
		return l
	context = search(query)
	try:
		reply = get_response(query,context)
		st.write(reply)
	except Exception as e:
		st.write(f"Got an error:\n{e}")
else:
	st.write("Answer displays here!")