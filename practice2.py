from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

input_ids = model.encode("The capital of France is").tolist()[0]

for i in range(5):
    logits = model.get_logits_from_input_ids(input_ids)

    next_token_id = logits.index(max(logits))

    input_ids.append(next_token_id)

    print(f"Step {i + 1}: {repr(model.decode(input_ids))}")