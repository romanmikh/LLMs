"""
Simply a wrapper around the Ollama interface API to use Llama as a local LLM server
Ollama is a local LLM server that can run various models, including Llama 3.2.
LLaMA (Large Language Model Meta AI) is a family of language models developed by Meta AI

Requirements:
1. Ollama (ollama run llama3.2)
2. Python 3.8+

Test if llama is running:
curl.exe -X POST http://localhost:11434/api/chat -H "Content-Type: application/json" --data-binary "@chat.json"

Run from dir behind LLMs as a module:
python -m LLMs.llama.llama
"""

import requests
from LLMs.utils import RED, CYA, YEL, GRE, BLU, MAG, RESET

API_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"  # match exactly with `$> ollama list`

def multiline_input(prompt="You: ") -> str:
    print(f"{YEL}{prompt}{RESET}", end="", flush=True)
    lines = []

    try:
        lines.append(input())
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
    except EOFError:
        pass  # graceful exit on Ctrl+D (Unix) or Ctrl+Z (Windows)

    return "\n".join(lines)


def chat_with_model():
    print(f"{MAG}Chat with Ollama model. Type 'exit' to quit.{RESET}")
    history = []

    while True:
        user_input = multiline_input("You: ")
        if user_input.lower() == "exit":
            break

        history.append({"role": "user", "content": user_input})

        payload = {
            "model": MODEL,
            "messages": history,
            "stream": False
        }

        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            data = response.json()

            reply = data["message"]["content"]
            print("GPT:", F"{CYA}{reply}{RESET}")
            history.append({"role": "assistant", "content": reply})

        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    chat_with_model()
