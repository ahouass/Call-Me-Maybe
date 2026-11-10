from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

input_ids = model.encode("Hello world")
ids = input_ids.tolist()[0]

print("Original:", model.decode(ids))
print("Token IDs:", ids)

logits = model.get_logits_from_input_ids(ids)

next_token_id = logits.index(max(logits))

print("Logits count:", len(logits))
print("Next token ID:", next_token_id)
print("Next token:", repr(model.decode([next_token_id])))

ids.append(next_token_id)

print("Complete text:", model.decode(ids))