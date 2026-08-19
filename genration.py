import retrieval as r


def generate_prompt(input_query):

    retrieved_knowledge = r.retrieve(input_query)

    instruction_prompt = f'''You are a helpful chatbot.

Use only the following pieces of context to answer the question.
Don't make up any new information:

{'\n'.join(
    [f' - {chunk}' for chunk, similarity in retrieved_knowledge]
)}
'''

    return instruction_prompt