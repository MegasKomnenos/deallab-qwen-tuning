import requests
import sys
import os
import json
import logging
import traceback
import argparse
import re
import time
import random

# Try to import readline for input history (Up arrow support)
try:
    import readline
except ImportError:
    pass

# -------------------------------------------------------------------------
# AESTHETICS & CONFIGURATION
# -------------------------------------------------------------------------

class Colors:
    ORANGE = "\033[38;5;208m" 
    GREEN = "\033[92m"        
    DIM = "\033[2m"           
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CLEAR_LINE = "\033[K"

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# --- CONFIGURATION ---
KSERVE_ENDPOINT = "http://localhost:8080/v1/chat/completions"
HOST_HEADER = "qwen-chat-kubeflow-user-example-com.example.com"

HEADERS = {
    "Host": HOST_HEADER,
    "Content-Type": "application/json"
}

THINK_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def slow_print(text, delay=0.01, color=Colors.RESET):
    sys.stdout.write(color)
    pos = 0
    while pos < len(text):
        match = ANSI_ESCAPE.search(text, pos)
        if match:
            start, end = match.span()
            for char in text[pos:start]:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(delay + random.uniform(0, 0.002))
            sys.stdout.write(match.group())
            sys.stdout.flush()
            pos = end
        else:
            for char in text[pos:]:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(delay + random.uniform(0, 0.002))
            break
    sys.stdout.write(Colors.RESET + "\n")

def print_banner():
    os.system('cls' if os.name == 'nt' else 'clear')
    banner = f"""
{Colors.GREEN}
   _______  _______  _______  _______  _______  _______ 
  |       ||       ||       ||       ||       ||       |
  |   _   ||  _____||_     _||   _   ||   _   ||   _   |
  |  | |  || |_____   |   |  |  | |  ||  |_|  ||  | |  |
  |  |_|  ||_____  |  |   |  |  |_|  ||       ||  |_|  |
  |       | _____| |  |   |  |       ||   _   ||       |
  |_______||_______|  |___|  |_______||__| |__||_______|
{Colors.DIM}  :: QWEN INTERFACE TERMINAL v1.2 :: SYNCED ::{Colors.RESET}
    """
    print(banner)

# -------------------------------------------------------------------------
# LOGIC
# -------------------------------------------------------------------------

class InMemorySession:
    def __init__(self, system_prompt="", show_thoughts=False):
        # FIX: Allow initializing with a specific persona
        self.history = [
            {"role": "system", "content": system_prompt}
        ]
        self.show_thoughts = show_thoughts
        
        print(f"{Colors.DIM}[System] Initializing connection to neural core...{Colors.RESET}")
        print(f"{Colors.DIM}[System] Endpoint: {KSERVE_ENDPOINT}{Colors.RESET}")
        print(f"{Colors.DIM}[System] Thoughts: {'VISIBLE' if show_thoughts else 'SUPPRESSED'}{Colors.RESET}")
        if system_prompt:
            print(f"{Colors.DIM}[System] Persona:  ACTIVE{Colors.RESET}\n")
        else:
            print(f"{Colors.DIM}[System] Persona:  DEFAULT{Colors.RESET}\n")

    def send_message(self, user_text: str):
        self.history.append({"role": "user", "content": user_text})

        # FIX: Added repetition_penalty and top_p to prevent loops and copying
        payload = {
            "model": "qwen",
            "messages": self.history,
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9,             # Added: Nucleus sampling for diversity
            "repetition_penalty": 1.1, # Added: Crucial fix for the "Demon" loop
            "stream": False
        }

        try:
            loading_msg = f"{Colors.GREEN}Qwen > {Colors.DIM}[Receiving data...]{Colors.RESET}"
            sys.stdout.write(loading_msg)
            sys.stdout.flush()
            
            response = requests.post(
                KSERVE_ENDPOINT, 
                headers=HEADERS, 
                json=payload
            )
            
            sys.stdout.write("\r" + Colors.CLEAR_LINE)
            sys.stdout.flush()

            if not response.text.strip():
                return f"{Colors.DIM}Error: Server returned {response.status_code} but BODY IS EMPTY.{Colors.RESET}"
            
            try:
                data = response.json()
            except json.JSONDecodeError:
                return f"{Colors.DIM}Error: Invalid JSON (Status {response.status_code}).{Colors.RESET}"
            
            if 'error' in data:
                err = data['error']
                msg = err.get('message', str(err)) if isinstance(err, dict) else str(err)
                return f"{Colors.DIM}API Error: {msg}{Colors.RESET}"

            if 'choices' in data:
                full_content = data['choices'][0]['message']['content']
                self.history.append({"role": "assistant", "content": full_content})
                
                if not self.show_thoughts:
                    display_content = THINK_PATTERN.sub('', full_content).lstrip()
                else:
                    display_content = full_content.replace("<think>", f"{Colors.DIM}<think>").replace("</think>", f"</think>{Colors.RESET}{Colors.GREEN}")

                return display_content
            
            return f"Error: Unexpected response format: {data.keys()}"

        except requests.exceptions.ConnectionError:
            sys.stdout.write("\r" + Colors.CLEAR_LINE)
            return f"{Colors.DIM}Error: Connection Refused. Is kubectl port-forward running?{Colors.RESET}"
        except Exception:
            sys.stdout.write("\r" + Colors.CLEAR_LINE)
            return traceback.format_exc()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Qwen CLI Interface")
    parser.add_argument(
        '--show-thoughts', 
        action='store_true', 
        default=False, 
        help='If set, displays the <think> process of the model.'
    )
    # FIX: Added arg for system prompt
    parser.add_argument(
        '--system-prompt',
        type=str,
        default="",
        help='The system persona (e.g., "You are a Victorian novelist").'
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    print_banner()
    
    # Pass the system prompt from CLI args
    session = InMemorySession(
        system_prompt=args.system_prompt, 
        show_thoughts=args.show_thoughts
    )
    
    print(f"{Colors.DIM}Type 'exit' or 'quit' to end session.{Colors.RESET}\n")

    while True:
        try:
            user_input = input(f"{Colors.ORANGE}User > {Colors.RESET}")
            
            if user_input.lower() in ["exit", "quit"]:
                print(f"\n{Colors.DIM}[System] Terminating Link... Goodbye.{Colors.RESET}")
                break
            
            if not user_input.strip():
                continue
            
            response_text = session.send_message(user_input)
            
            sys.stdout.write(f"{Colors.GREEN}Qwen > {Colors.RESET}")
            speed = 0.02 if len(response_text) < 200 else 0.005
            slow_print(response_text, delay=speed, color=Colors.GREEN)
            print() 
            
        except KeyboardInterrupt:
            print(f"\n{Colors.DIM}[System] Interrupt detected. Shutting down.{Colors.RESET}")
            break

if __name__ == "__main__":
    main()