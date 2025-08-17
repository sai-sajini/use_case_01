
# llm.py: Wrapper for calling OpenRouter API (GPT-4 Turbo) or other OpenAI-compatible endpoints
import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

# --- OpenRouter API (GPT-4 Turbo or other) ---
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_API_BASE = os.getenv('OPENROUTER_API_BASE', 'https://openrouter.ai/api/v1')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'openai/gpt-oss-20b:free')

def call_openrouter(prompt, model=OPENROUTER_MODEL):
	"""Call OpenRouter API and return the response text."""
	if not OPENROUTER_API_KEY:
		raise RuntimeError("OPENROUTER_API_KEY not set in .env")
	print(f"[call_openrouter] Calling OpenRouter. Model: {model}\nPrompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
	headers = {
		"Authorization": f"Bearer {OPENROUTER_API_KEY}",
		"Content-Type": "application/json"
	}
	data = {
		"model": model,
		"messages": [
			{"role": "user", "content": prompt}
		]
	}
	response = requests.post(f"{OPENROUTER_API_BASE}/chat/completions", headers=headers, data=json.dumps(data))
	if response.status_code != 200:
		print(f"[call_openrouter] OpenRouter error: {response.text}")
		raise RuntimeError(f"OpenRouter error: {response.text}")
	result = response.json()
	output = result['choices'][0]['message']['content'].strip()
	print(f"[call_openrouter] OpenRouter output: {output}")
	return output



def llm_suggest_category_name(tickets):
	prompt = (
		"You are an AI that helps categorize IT incident tickets concisely."
		"Given the following ticket(s):"
		f"{tickets}"
		"Respond as follows:"
		"- If you can confidently assign a category, reply with a short, specific category name (maximum 3 words, no quotes, no extra text)."
		"- If you CANNOT confidently assign a category, reply with exactly this word: Uncategorized"
		"- Do NOT explain, apologize, repeat the prompt, or add any other information."
		"- Reply with only the category name or 'Uncategorized', nothing else."
		"Examples:"
		"Correct: Network Issue"
		"Correct: Performance"
		"Correct: Uncategorized"
		"Incorrect: Sorry, I cannot categorize this."
		"Incorrect: This ticket seems to be about..."
		"Incorrect: 'Network Issue'"
		"Incorrect: Category: Network Issue"
		"Now, what is the best category for these tickets?"
	)
	print(f"[llm_suggest_category_name] Tickets: {tickets}")
	return call_openrouter(prompt)

def llm_merge_decision(name_a, examples_a, name_b, examples_b):
	prompt = f"Category A: {name_a}, Tickets: {examples_a} | Category B: {name_b}, Tickets: {examples_b}. Should these be merged? Respond YES or NO."
	print(f"[llm_merge_decision] A: {name_a}, B: {name_b}")
	return call_openrouter(prompt)

def llm_adjust_threshold(current_threshold, num_categories, avg_tickets_per_category):
	prompt = f"Current similarity threshold: {current_threshold}. Categories: {num_categories}, Avg tickets/category: {avg_tickets_per_category}. Respond with INCREASE / DECREASE / KEEP."
	print(f"[llm_adjust_threshold] Threshold: {current_threshold}, Categories: {num_categories}, Avg: {avg_tickets_per_category}")
	return call_openrouter(prompt)

# --- DeepSeek R1 via Ollama (commented) ---
# import subprocess
# def call_ollama(prompt, model='deepseek-coder:latest'):
#     """Call Ollama with DeepSeek R1 and return the response text."""
#     result = subprocess.run([
#         'ollama', 'run', model
#     ], input=prompt.encode('utf-8'), capture_output=True)
#     if result.returncode != 0:
#         raise RuntimeError(f"Ollama error: {result.stderr.decode('utf-8')}")
#     return result.stdout.decode('utf-8').strip()

# def llm_suggest_category_name(tickets):
#     prompt = f"You are an AI helping categorize IT incident tickets. Given the following tickets: {tickets}, suggest a concise category name under 3 words(strictly under 3 words. No extra details or reasoning needed. respond with only the category name). If you cannot find a suitable category, respond with only the word 'Uncategorized' without any quotes."
#     return call_ollama(prompt)

# def llm_merge_decision(name_a, examples_a, name_b, examples_b):
#     prompt = f"Category A: {name_a}, Tickets: {examples_a} | Category B: {name_b}, Tickets: {examples_b}. Based on the given category names and tickets, should these be merged? Respond only with YES or NO. Nothing more, nothing less."
#     return call_ollama(prompt)

# def llm_adjust_threshold(current_threshold, num_categories, avg_tickets_per_category):
#     prompt = f"Current similarity threshold is: {current_threshold}. number of Categories is: {num_categories}, Avg tickets per category is: {avg_tickets_per_category}. Based on given data, what should we do? Respond only with INCREASE / DECREASE / KEEP. No extra details or reasoning needed."
#     return call_ollama(prompt)

# --- DeepSeek R1 via GitHub Marketplace API Key ---
# import os
# from azure.ai.inference import ChatCompletionsClient
# from azure.ai.inference.models import UserMessage
# from azure.core.credentials import AzureKeyCredential

# client = ChatCompletionsClient(
# 	endpoint="https://models.github.ai/inference",
# 	credential=AzureKeyCredential(os.environ.get("GITHUB_TOKEN", "")),
# )

# def call_github_deepseek(prompt, max_tokens=2048):
# 	response = client.complete(
# 		messages=[UserMessage(prompt)],
# 		model="deepseek/DeepSeek-R1",
# 		max_tokens=max_tokens,
# 	)
# 	return response.choices[0].message.content

# def llm_suggest_category_name(tickets):
# 	prompt = f"You are an AI helping categorize IT incident tickets. Given the following tickets: {tickets}, suggest a concise category name under 3 words(strictly under 3 words. No extra details or reasoning needed. respond with only the category name). If you cannot find a suitable category, respond with only the word 'Uncategorized' without any quotes. Output only the category name, e.g. Network Issue, Performance, or Uncategorized. Incorrect: <think>, Incorrect: <CategoryName>."
# 	return call_github_deepseek(prompt)

# def llm_merge_decision(name_a, examples_a, name_b, examples_b):
# 	prompt = f"Category A: {name_a}, Tickets: {examples_a} | Category B: {name_b}, Tickets: {examples_b}. Based on the given category names and tickets, should these be merged? Respond only with YES or NO. Nothing more, nothing less."
# 	return call_github_deepseek(prompt)

# def llm_adjust_threshold(current_threshold, num_categories, avg_tickets_per_category):
# 	prompt = f"Current similarity threshold is: {current_threshold}. number of Categories is: {num_categories}, Avg tickets per category is: {avg_tickets_per_category}. Based on given data, what should we do? Respond only with INCREASE / DECREASE / KEEP. No extra details or reasoning needed."
# 	return call_github_deepseek(prompt)
