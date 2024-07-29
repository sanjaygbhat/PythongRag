from app import llama_call

test_messages = [
    {
        "role": "user",
        "content": "Hello!"
    }
]

try:
    result = llama_call(test_messages)
    print(f"Result: {result}")
except Exception as e:
    print(f"An error occurred: {e}")