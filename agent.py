# agent.py: Main agent loop orchestrating everything


import os
import time
from memory import load_memory, save_memory, clear_memory_file
from categorization import load_tickets, assign_ticket_to_category, create_category, merge_categories, rename_category
from embeddings import get_embedding
from llm import llm_suggest_category_name, llm_merge_decision
from decision import adjust_threshold, decide_next_action
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

DATA_DIR = 'data'
TICKETS_FILE = os.path.join(DATA_DIR, 'tickets.xlsx')
MEMORY_FILE = os.path.join(DATA_DIR, 'category_memory.json')
OUTPUT_FILE = os.path.join(DATA_DIR, 'tickets_categorized.xlsx')


def select_text_column(df):
	print("[select_text_column] Called. Columns:", df.columns)
	# If 'Description' exists, use it
	if 'Description' in df.columns:
		print("[select_text_column] Using 'Description' column.")
		return 'Description'
	# Otherwise, pick the text column with the highest mean text length
	text_cols = [col for col in df.columns if df[col].dtype == object]
	print(f"[select_text_column] Text columns found: {text_cols}")
	if not text_cols:
		raise ValueError('No text columns found in the tickets file.')
	best_col = None
	best_len = -1
	for col in text_cols:
		texts = df[col].astype(str).fillna('')
		mean_len = texts.str.len().mean()
		print(f"[select_text_column] Checking column: {col}, mean length: {mean_len}")
		if mean_len > best_len:
			best_len = mean_len
			best_col = col
	if best_col is None:
		raise ValueError('No suitable text column found for ticket description.')
	print(f"[select_text_column] Selected column: {best_col}")
	return best_col

def main():
	# Clear the memory file at the start of each run
	clear_memory_file(MEMORY_FILE)
	start_time = time.time()
	print("[main] Loading tickets from:", TICKETS_FILE)
	tickets_df = load_tickets(TICKETS_FILE)
	print("[main] Tickets loaded. Shape:", tickets_df.shape)
	text_col = select_text_column(tickets_df)
	print(f"[main] Using text column: {text_col}")
	memory = load_memory(MEMORY_FILE)
	print(f"[main] Loaded memory. Categories: {list(memory.get('categories', {}).keys())}")
	threshold = 0.75
	print(f"[main] Initial threshold: {threshold}")
	tickets_df['Category'] = None
	unprocessed = set(tickets_df.index)
	print(f"[main] Unprocessed tickets: {len(unprocessed)}")

	while unprocessed or optimizations_possible(memory):
		print(f"[main] Loop start. Unprocessed: {len(unprocessed)}")
		state = {
			'unprocessed_tickets': bool(unprocessed),
			'need_create': False,
			'can_merge': False,
			'can_rename': False
		}
		# Check for possible merges (simple pairwise check)
		cats = list(memory.get('categories', {}).keys())
		for i in range(len(cats)):
			for j in range(i+1, len(cats)):
				name_a, name_b = cats[i], cats[j]
				ex_a = memory['categories'][name_a]['examples'][:2]
				ex_b = memory['categories'][name_b]['examples'][:2]
				print(f"[main] Checking merge: {name_a} <-> {name_b}")
				resp = llm_merge_decision(name_a, ex_a, name_b, ex_b)
				print(f"[main] Merge decision response: {resp}")
				if 'YES' in resp.upper():
					print(f"[main] Merge possible: {name_a}, {name_b}")
					state['can_merge'] = (name_a, name_b)
					break
			if state['can_merge']:
				break
		# Check for possible renames (not implemented, placeholder)
		# ...

		print(f"[main] State before decision: {state}")
		action = decide_next_action(state)
		print(f"[main] Decided action: {action}")

		if action == 'assign' and unprocessed:
			idx = unprocessed.pop()
			ticket = str(tickets_df.loc[idx, text_col])
			summary_value = str(tickets_df.loc[idx, 'Summary*']) if 'Summary*' in tickets_df.columns else ticket
			print(f"[main] Assigning ticket idx {idx}: {ticket}")
			cat = assign_ticket_to_category(ticket, memory, threshold, get_embedding)
			print(f"[main] Assigned to category: {cat}")
			if cat:
				tickets_df.at[idx, 'Category'] = cat
				memory['categories'][cat]['examples'].append(summary_value)
			else:
				print(f"[main] No suitable category found. Creating new category immediately.")
				llm_response, ticket_emb = create_category(ticket, memory, get_embedding, llm_suggest_category_name)
				print(f"[main] LLM response: {llm_response}")
				cat_name = llm_response.strip().splitlines()[0]
				if 'uncategorized' in cat_name.lower():
					print(f"[main] LLM could not suggest a category. Assigning default: Uncategorized")
					cat_name = 'Uncategorized'
				print(f"[main] Created category: {cat_name}")
				tickets_df.at[idx, 'Category'] = cat_name
				# Update memory with new category, storing summary in examples
				if 'categories' not in memory:
					memory['categories'] = {}
				memory['categories'][cat_name] = {'examples': [summary_value], 'embedding': ticket_emb}
		elif action == 'create' and unprocessed:
			idx = unprocessed.pop()
			ticket = str(tickets_df.loc[idx, text_col])
			summary_value = str(tickets_df.loc[idx, 'Summary*']) if 'Summary*' in tickets_df.columns else ticket
			print(f"[main] Creating new category for ticket idx {idx}: {ticket}")
			llm_response, ticket_emb = create_category(ticket, memory, get_embedding, llm_suggest_category_name)
			print(f"[main] LLM response: {llm_response}")
			cat_name = llm_response.strip().splitlines()[0]
			if 'uncategorized' in cat_name.lower():
				print(f"[main] LLM could not suggest a category. Assigning default: Uncategorized")
				cat_name = 'Uncategorized'
			print(f"[main] Created category: {cat_name}")
			tickets_df.at[idx, 'Category'] = cat_name
			if 'categories' not in memory:
				memory['categories'] = {}
			memory['categories'][cat_name] = {'examples': [summary_value], 'embedding': ticket_emb}
		elif action == 'merge' and state['can_merge']:
			cat_a, cat_b = state['can_merge']
			print(f"[main] Merging categories: {cat_a}, {cat_b}")
			merge_categories(cat_a, cat_b, memory)
		elif action == 'rename' and state['can_rename']:
			# Placeholder for rename logic
			print(f"[main] Rename action selected, but not implemented.")
			pass
		elif action == 'adjust_threshold':
			print(f"[main] Adjusting threshold. Current: {threshold}")
			threshold = adjust_threshold(memory, threshold)
			print(f"[main] New threshold: {threshold}")

		print(f"[main] Saving memory and tickets.")
		save_memory(MEMORY_FILE, memory)
		tickets_df.to_excel(OUTPUT_FILE, index=False)

	print(f"Categorization complete. Output: {OUTPUT_FILE}\nUpdated memory: {MEMORY_FILE}")
	end_time = time.time()
	elapsed = end_time - start_time
	elapsed_minutes = elapsed / 60
	print(f"[main] Total time taken: {elapsed_minutes:.2f} minutes")

def optimizations_possible(memory):
	# Placeholder: could check for merge/rename opportunities
	return False

if __name__ == '__main__':
	main()
