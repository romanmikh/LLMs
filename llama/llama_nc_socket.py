"""
Simply a wrapper around the Ollama interface API to use Llama as a local LLM server
Ollama is a local LLM server that can run various models, including Llama 3.2.
LLaMA (Large Language Model Meta AI) is a family of language models developed by Meta AI

Requirements:
1. Ollama (ollama run llama3.2)
2. Python 3.8+
3. sudo ufw allow 12345
4. https://portchecker.co/check-v0 <-- check if port 12345 is open

Test if llama is running:
curl.exe -X POST http://localhost:11434/api/chat -H "Content-Type: application/json" --data-binary "@chat.json"

Run from dir behind LLMs as a module:
python -m LLMs.llama.llama


If running on fresh machine:
sudo apt update
sudo apt install curl
sudo snap install ollama
ollama run llama3.2 # loop processm, run in separate terminal

venv:
python3 -m venv myenv
source myenv/bin/activate
pip install requests

"""

import socket
import requests
from LLMs.utils import RED, CYA, YEL, GRE, BLU, MAG, RESET

API_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"  # match exactly with `$> ollama list`
HOST = "0.0.0.0"
PORT = 12345

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
    print(f"{MAG}Waiting for mobile connection on port {PORT}...{RESET}")
    history = []

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        conn, addr = s.accept()
        print(f"{GRE}Connected from {addr}{RESET}")

        with conn:
            while True:
                conn.sendall(b"You: ")
                data = b""
                while True:
                    chunk = conn.recv(1024)
                    if not chunk or b"\n" in chunk:
                        data += chunk
                        break
                    data += chunk

                user_input = data.decode().strip()  # ← FIXED: now outside inner loop

                if user_input.lower() == "exit":
                    conn.sendall(b"Goodbye!\n")
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
                    reply = response.json()["message"]["content"]

                    history.append({"role": "assistant", "content": reply})
                    conn.sendall(f"\nGPT: {reply}\n\n".encode())

                except requests.exceptions.HTTPError:
                    conn.sendall(b"HTTP error from Ollama.\n")
                except Exception as e:
                    conn.sendall(f"Error: {e}\n".encode())

if __name__ == "__main__":
    chat_with_model()
