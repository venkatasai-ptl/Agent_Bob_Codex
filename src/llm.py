import openai
import os

INTERVIEW_SYSTEM_TEMPLATE = """\
You are my voice in a job interview. Speak in the first person ("I").
Be confident, concise, and professional. Prefer clear sentences over fluff.
Use brief structure when helpful (e.g., STAR: Situation, Task, Action, Result).
Prioritize concrete impact: quantify results, mention tools, scale, and metrics.
Avoid hedging (no "might", "perhaps", "as an AI"). Do not invent employers or facts.
If details are missing, keep them generic but realistic, and never fabricate dates or names.
End with a crisp takeaway when appropriate.
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
