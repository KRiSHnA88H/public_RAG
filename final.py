import genration as g
import ollama
import embedding as e

stream = ollama.chat(
  model=e.LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': g.instruction_prompt},
    {'role': 'user', 'content': g.input_query},
  ],
  stream=True,
)

# print the response from the chatbot in real-time
print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
