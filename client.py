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
    pass  # Windows users might need 'pyreadline3' or just ignore this

# -------------------------------------------------------------------------
# AESTHETICS & CONFIGURATION
# -------------------------------------------------------------------------

# ANSI Colors for CRT Monitor Feel
class Colors:
    ORANGE = "\033[38;5;208m" 
    GREEN = "\033[92m"        
    DIM = "\033[2m"           
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CLEAR_LINE = "\033[K" # ANSI code to clear text from cursor to end of line

# Silence internal request logs
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# --- CONFIGURATION ---
KSERVE_ENDPOINT = "http://localhost:8080/v1/chat/completions"
# CRITICAL: This must match the URL shown in `kubectl get isvc`
HOST_HEADER = "qwen-chat-kubeflow-user-example-com.example.com"

HEADERS = {
    "Host": HOST_HEADER,
    "Content-Type": "application/json"
}

# Regex to find thinking blocks and ANSI codes
THINK_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def slow_print(text, delay=0.01, color=Colors.RESET):
    """
    Simulates old terminal typing effect.
    Improvements: Now detects ANSI codes so it doesn't print color codes character-by-character.
    """
    sys.stdout.write(color)
    
    # Split text into (ansi_code, regular_text) chunks
    # This logic prevents the typewriter effect from breaking hidden color codes
    pos = 0
    while pos < len(text):
        match = ANSI_ESCAPE.search(text, pos)
        if match:
            # Print text before the code slowly
            start, end = match.span()
            for char in text[pos:start]:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(delay + random.uniform(0, 0.002))
            
            # Print the ANSI code instantly (invisible to user but changes state)
            sys.stdout.write(match.group())
            sys.stdout.flush()
            pos = end
        else:
            # No more colors, print rest of string
            for char in text[pos:]:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(delay + random.uniform(0, 0.002))
            break
            
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
{Colors.DIM}  :: QWEN INTERFACE TERMINAL v1.1 :: SYNCED ::{Colors.RESET}
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
        time.sleep(0.3)
        print(f"{Colors.DIM}[System] Endpoint: {KSERVE_ENDPOINT}{Colors.RESET}")
        print(f"{Colors.DIM}[System] Host:     {HEADERS['Host']}{Colors.RESET}")
        print(f"{Colors.DIM}[System] Thoughts: {'VISIBLE' if show_thoughts else 'SUPPRESSED'}{Colors.RESET}\n")

    def send_message(self, user_text: str):
        self.history.append({"role": "user", "content": user_text})

        payload = {
            "model": "qwen",
            "messages": self.history,
            "max_tokens": 4096,
            "temperature": 0.7,
            "stream": False
        }

        try:
            # 1. Send Request with Loading Indicator
            # \r moves cursor to start of line to allow overwriting later
            loading_msg = f"{Colors.GREEN}Qwen > {Colors.DIM}[Receiving data...]{Colors.RESET}"
            sys.stdout.write(loading_msg)
            sys.stdout.flush()
            
            response = requests.post(
                KSERVE_ENDPOINT, 
                headers=HEADERS, 
                json=payload
            )
            
            # 2. Clear the loading message
            # \r moves to start, CLEAR_LINE wipes the text
            sys.stdout.write("\r" + Colors.CLEAR_LINE)
            sys.stdout.flush()

            # 3. Validation
            if not response.text.strip():
                return f"{Colors.DIM}Error: Server returned {response.status_code} but BODY IS EMPTY.{Colors.RESET}"
            
            try:
                data = response.json()
            except json.JSONDecodeError:
                return f"{Colors.DIM}Error: Invalid JSON (Status {response.status_code}).{Colors.RESET}"
            
            # 4. Error Handling
            if 'error' in data:
                err = data['error']
                msg = err.get('message', str(err)) if isinstance(err, dict) else str(err)
                return f"{Colors.DIM}API Error: {msg}{Colors.RESET}"

            if 'message' in data and 'code' in data:
                return f"{Colors.DIM}API Error: {data['message']} (Code: {data['code']}){Colors.RESET}"

            # 5. Success
            if 'choices' in data:
                full_content = data['choices'][0]['message']['content']
                self.history.append({"role": "assistant", "content": full_content})
                
                if not self.show_thoughts:
                    # Remove <think> blocks for clean output
                    display_content = THINK_PATTERN.sub('', full_content).lstrip()
                else:
                    # Dim the think blocks for style
                    display_content = full_content.replace("<think>", f"{Colors.DIM}<think>").replace("</think>", f"</think>{Colors.RESET}{Colors.GREEN}")

                return display_content
            
            return f"Error: Unexpected response format: {data.keys()}"

        except requests.exceptions.ConnectionError:
            sys.stdout.write("\r" + Colors.CLEAR_LINE) # Clear loading text on error
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
    return parser.parse_args()

def main():
    args = parse_arguments()
    print_banner()
    
    session = InMemorySession(show_thoughts=args.show_thoughts)
    
    print(f"{Colors.DIM}Type 'exit' or 'quit' to end session.{Colors.RESET}\n")

    while True:
        try:
            # 1. User Input
            # Using standard input() handles the prompt color simulation
            user_input = input(f"{Colors.ORANGE}User > {Colors.RESET}")
            
            if user_input.lower() in ["exit", "quit"]:
                print(f"\n{Colors.DIM}[System] Terminating Link... Goodbye.{Colors.RESET}")
                break
            
            if not user_input.strip():
                continue
            
            # 2. Get Response
            # Note: We do NOT print "Qwen >" here anymore. 
            # The session.send_message function handles the temporary "Loading..." label.
            response_text = session.send_message(user_input)
            
            # 3. Print Final Output
            # Now we print the permanent label + the response
            sys.stdout.write(f"{Colors.GREEN}Qwen > {Colors.RESET}")
            
            # Calculate dynamic delay: shorter delay for long text
            speed = 0.02 if len(response_text) < 200 else 0.005
            slow_print(response_text, delay=speed, color=Colors.GREEN)
            print() # Extra spacing
            
        except KeyboardInterrupt:
            print(f"\n{Colors.DIM}[System] Interrupt detected. Shutting down.{Colors.RESET}")
            break

if __name__ == "__main__":
    main()