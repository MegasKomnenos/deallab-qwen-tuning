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

# -------------------------------------------------------------------------
# AESTHETICS & CONFIGURATION
# -------------------------------------------------------------------------

# ANSI Colors for CRT Monitor Feel
class Colors:
    # 38;5;208 is a standard Xterm 256-color code for Orange
    ORANGE = "\033[38;5;208m" 
    # Bright Green for that classic phosphorus glow
    GREEN = "\033[92m"        
    # Dim Grey for system messages
    DIM = "\033[2m"           
    RESET = "\033[0m"
    BOLD = "\033[1m"

# Silence internal request logs
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Update this to match your specific port-forward
KSERVE_ENDPOINT = "http://localhost:8080/v1/chat/completions"

HEADERS = {
    # CRITICAL: This must match the URL shown in `kubectl get isvc`
    "Host": "qwen-chat-kubeflow-user-example-com.example.com",
    "Content-Type": "application/json"
}

# Regex to find thinking blocks
THINK_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)

def slow_print(text, delay=0.01, color=Colors.RESET):
    """Simulates old terminal typing effect."""
    sys.stdout.write(color)
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        # Randomize delay slightly for realism
        time.sleep(delay + random.uniform(0, 0.005)) 
    sys.stdout.write(Colors.RESET + "\n")

def print_banner():
    """Prints a retro ASCII banner."""
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
{Colors.DIM}  :: QWEN INTERFACE TERMINAL v1.0 :: SYNCED ::{Colors.RESET}
    """
    print(banner)

# -------------------------------------------------------------------------
# LOGIC
# -------------------------------------------------------------------------

class InMemorySession:
    def __init__(self, show_thoughts=False):
        self.history = [
            {"role": "system", "content": ""}
        ]
        self.show_thoughts = show_thoughts
        
        # System boot-up effect
        print(f"{Colors.DIM}[System] Initializing connection to neural core...{Colors.RESET}")
        time.sleep(0.5)
        print(f"{Colors.DIM}[System] Endpoint: {KSERVE_ENDPOINT}{Colors.RESET}")
        print(f"{Colors.DIM}[System] Host Header: {HEADERS['Host']}{Colors.RESET}")
        print(f"{Colors.DIM}[System] Thought Stream: {'VISIBLE' if show_thoughts else 'SUPPRESSED'}{Colors.RESET}\n")

    def send_message(self, user_text: str):
        self.history.append({"role": "user", "content": user_text})

        payload = {
            "model": "qwen_merged_final",
            "messages": self.history,
            "max_tokens": 4096,
            "temperature": 0.7,
            "stream": False
        }

        try:
            # 1. Send Request
            sys.stdout.write(f"{Colors.GREEN}Qwen > {Colors.DIM}[Receiving data...]{Colors.RESET}\r")
            sys.stdout.flush()
            
            response = requests.post(
                KSERVE_ENDPOINT, 
                headers=HEADERS, 
                json=payload
            )
            
            sys.stdout.write(" " * 50 + "\r") 

            # 2. Debugging Block
            if not response.text.strip():
                return f"{Colors.DIM}Error: Server returned {response.status_code} but BODY IS EMPTY.{Colors.RESET}"
            
            # 3. Parse JSON safely
            try:
                data = response.json()
            except json.JSONDecodeError:
                return f"{Colors.DIM}Error: Invalid JSON received (Status {response.status_code}).{Colors.RESET}"
            
            # ---------------------------------------------------------
            # NEW: ROBUST ERROR HANDLING
            # ---------------------------------------------------------
            
            # Case A: Standard OpenAI Error (Nested)
            # { "error": { "message": "..." } }
            if 'error' in data:
                err_content = data['error']
                if isinstance(err_content, dict):
                    msg = err_content.get('message', str(err_content))
                else:
                    msg = str(err_content)
                return f"{Colors.DIM}API Error: {msg}{Colors.RESET}"

            # Case B: Flattened Error (The one you are hitting)
            # { "message": "...", "type": "...", "code": "..." }
            if 'message' in data and 'code' in data:
                return f"{Colors.DIM}API Error: {data['message']} (Code: {data['code']}){Colors.RESET}"

            # Case C: Success
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
            return f"{Colors.DIM}Error: Connection Refused at {KSERVE_ENDPOINT}. Is kubectl port-forward running?{Colors.RESET}"
        except Exception as e:
            return traceback.format_exc()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Qwen CLI Interface")
    parser.add_argument(
        '--show-thoughts', 
        action='store_true', 
        default=False, 
        help='If set, displays the <think> process of the model.'
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    print_banner()
    
    session = InMemorySession(show_thoughts=args.show_thoughts)
    
    print(f"{Colors.DIM}Type 'exit' or 'quit' to end session.{Colors.RESET}\n")

    while True:
        try:
            # Styled User Prompt (Orange)
            # We use standard input but simulate color by printing the prompt part first
            user_input = input(f"{Colors.ORANGE}User > {Colors.RESET}")
            
            if user_input.lower() in ["exit", "quit"]:
                print(f"\n{Colors.DIM}[System] Terminating Link... Goodbye.{Colors.RESET}")
                break
            
            # Print label for Bot
            sys.stdout.write(f"{Colors.GREEN}Qwen > {Colors.RESET}")
            sys.stdout.flush()
            
            # Get response
            response_text = session.send_message(user_input)
            
            # Typewriter effect for the response
            # Calculate dynamic delay: shorter delay for long text to save time
            speed = 0.02 if len(response_text) < 200 else 0.005
            slow_print(response_text, delay=speed, color=Colors.GREEN)
            print() # Extra spacing
            
        except KeyboardInterrupt:
            print(f"\n{Colors.DIM}[System] Interrupt detected. Shutting down.{Colors.RESET}")
            break

if __name__ == "__main__":
    main()
