tokens = ["hello", "12", "true", "45", "-5", "3.14"]
logits = [9.0, 6.0, 8.0, 5.0, 7.0, 10.0]


def is_allowed(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


allowed_ids = []

for token_id, token in enumerate(tokens):
    if is_allowed(token):
        allowed_ids.append(token_id)

print("Allowed IDs:", allowed_ids)

next_id = max(
    allowed_ids,
    key=lambda i: logits[i]
)

print("Selected ID:", next_id)
print("Selected token:", tokens[next_id])