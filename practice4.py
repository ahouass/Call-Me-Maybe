import json

from llm_sdk import Small_LLM_Model


with open(
    "data/input/functions_definition.json",
    "r",
    encoding="utf-8"
) as file:
    functions = json.load(file)

user_prompt = "What is the sum of 2 and 3?"

llm_prompt = f"""
You are a function calling assistant.

Available functions:
{json.dumps(functions, indent=2)}

User request:
{user_prompt}

Choose the appropriate function and extract its arguments.

Return only JSON with the following structure:
{{"name": "function_name", "parameters": {{}}}}

JSON:
"""

model = Small_LLM_Model()

input_ids = model.encode(llm_prompt).tolist()[0]

generated_ids = []

for i in range(100):
    logits = model.get_logits_from_input_ids(input_ids)

    next_token_id = logits.index(max(logits))

    input_ids.append(next_token_id)
    generated_ids.append(next_token_id)

    generated_text = model.decode(generated_ids)

    try:
        result = json.loads(generated_text)
    except json.JSONDecodeError:
        continue

    if isinstance(result, dict):
        print("Complete JSON detected!")
        break

print(json.dumps(result, indent=2))

print("\n\nFinal output:")
print(model.decode(generated_ids))