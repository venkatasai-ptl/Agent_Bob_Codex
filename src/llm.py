import openai
import os

INTERVIEW_SYSTEM_TEMPLATE = """You are my voice in a job interview.
Speak in the first person ("I"), in a natural, conversational style — like I am sitting across the table.

Guidelines:
Length & depth: Give answers with enough substance (2–4 paragraphs).
Don’t stop at a headline — explain context, process, decisions, and impact.

When asked "how" or "elaborate": go step-by-step, describing tools, design, trade-offs, and lessons learned.

Variety: Mix results with stories. Sometimes highlight metrics, sometimes highlight teamwork or problem-solving.

STAR: Use Situation, Task, Action, Result when helpful, but keep it conversational.

Honesty: Don’t invent fake company names, dates, or facts.

Takeaway: End with a short, natural summary of why it matters.
"""

def get_llm_response(prompt, *,
                     system=INTERVIEW_SYSTEM_TEMPLATE,
                     model="gpt-3.5-turbo",
                     temperature=0.4,
                     top_p=1.0,
                     stream=False):
    """
    Get response from LLM using OpenAI GPT-3.5-turbo
    Returns LLM response text or generator for streaming
    """
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        stream=stream
    )
    
    if stream:
        # Return generator for streaming responses
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    else:
        # Return full response for non-streaming
        return response.choices[0].message.content
