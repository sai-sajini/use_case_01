# memory.py: Functions to load/save category memory as JSON
import json
import os

def load_memory(memory_path):
	"""Load category memory from a JSON file. If file does not exist, create it as empty and return empty dict."""
	print(f"[load_memory] Loading memory from: {memory_path}")
	if not os.path.exists(memory_path):
		print(f"[load_memory] File not found. Creating empty file.")
		with open(memory_path, 'w', encoding='utf-8') as f:
			json.dump({}, f, indent=2, ensure_ascii=False)
		return {}
	with open(memory_path, 'r', encoding='utf-8') as f:
		data = json.load(f)
		print(f"[load_memory] Loaded memory. Keys: {list(data.keys())}")
		return data

def save_memory(memory_path, memory):
	"""Save category memory to a JSON file."""
	print(f"[save_memory] Saving memory to: {memory_path}")
	with open(memory_path, 'w', encoding='utf-8') as f:
		json.dump(memory, f, indent=2, ensure_ascii=False)
	print(f"[save_memory] Memory saved.")

# Clear the memory file at the start of each run
def clear_memory_file(memory_path):
	print(f"[clear_memory_file] Clearing memory file: {memory_path}")
	with open(memory_path, 'w', encoding='utf-8') as f:
		json.dump({}, f, indent=2, ensure_ascii=False)
	print(f"[clear_memory_file] Memory file cleared.")
