import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai_object = OpenAI()

def text_embedding(txt) -> None:
  model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", token = token)
  return model.encode(txt)
def get_response(query, context):
  # context = generate_context(query, results = 20)

  system_prompt = '''
                    Every sentense in the context represent a information statement.
                    Forget everything and focus only on context.
                    Analyze the given context only and give answers only from that.
                    If you can not find the query related to context or query is empty, say "I don't know.".
                    Be concise and and include detailed response.
                    Fetch the references as well from the text and show it at the end.
                    During response generation summarize the answer with all the reviews available from text.
                  '''
  user_prompt = f'''
                  Based on context: {context}, Answer the below query : {query}
               '''

  system_role = {
    "role" : "system",
    "content" : system_prompt
  }

  user_role = {
      "role" : "user",
      "content" : user_prompt
  }

  response = openai_object.chat.completions.create(
                                      model = "gpt-3.5-turbo",
                                      messages = [system_role, user_role]
                                      )
  result = response.choices[0].message.content

  return result


