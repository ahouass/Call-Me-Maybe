def validate_simple_json(text: str) -> bool:
    state = "START"

    for char in text:
        if state == "START":
            if char == "{":
                state = "KEY"
            else:
                return False

        elif state == "KEY":
            if char == '"':
                state = "KEY_CONTENT"
            else:
                return False

        elif state == "KEY_CONTENT":
            if char == "a":
                state = "KEY_END"
            else:
                return False

        elif state == "KEY_END":
            if char == '"':
                state = "COLON"
            else:
                return False

        elif state == "COLON":
            if char == ":":
                state = "NUMBER"
            else:
                return False

        elif state == "NUMBER":
            if char.isascii() and char.isdigit():
                state = "CLOSE"
            else:
                return False

        elif state == "CLOSE":
            if char == "}":
                state = "DONE"
            else:
                return False

        else:
            return False

    return state == "DONE"

print(validate_simple_json('{"a":2}'))
print(validate_simple_json('{"a":"hello"}'))
print(validate_simple_json('{"b":2}'))