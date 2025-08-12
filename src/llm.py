import openai
import os

def get_llm_response(prompt):
    """
    Get response from LLM using OpenAI GPT-3.5-turbo
    Returns LLM response text
    """
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
