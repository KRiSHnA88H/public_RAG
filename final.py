import ollama
import embedding as e
import genration as g


def generate_answer(input_query):

    instruction_prompt = g.generate_prompt(input_query)

    stream = ollama.chat(
        model=e.LANGUAGE_MODEL,
        messages=[
            {
                'role': 'system',
                'content': instruction_prompt
            },
            {
                'role': 'user',
                'content': input_query
            }
        ],
        stream=True,
    )

    answer = ""

    for chunk in stream:

        answer += chunk['message']['content']

    return answer