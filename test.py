import ast
test_str = repr("\n\n{\n  \"user_need\": \"Update me\",\n  \"I1\": 1,\n  \"I3\": 3,\n  \"I4\": 3\n}")

print((ast.literal_eval(test_str).strip()))
