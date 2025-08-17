last_category_reason = None
# categorization.py: assign/create/merge/rename category logic
import pandas as pd
import numpy as np

def load_tickets(file_path):
	"""Load tickets from Excel file. Returns a pandas DataFrame."""
	print(f"[load_tickets] Loading tickets from: {file_path}")
	df = pd.read_excel(file_path)
	print(f"[load_tickets] Loaded {len(df)} tickets. Columns: {df.columns}")
	return df

def assign_ticket_to_category(ticket, memory, threshold, get_embedding):
	"""Assign ticket to best matching category if similarity > threshold. Returns category name or None."""
	print(f"[assign_ticket_to_category] Ticket: {ticket}")
	ticket_emb = get_embedding(ticket)
	best_cat = None
	best_sim = -1
	for cat, cat_data in memory.get('categories', {}).items():
		cat_emb = cat_data['embedding']
		sim = cosine_similarity(ticket_emb, cat_emb)
		print(f"[assign_ticket_to_category] Category: {cat}, Similarity: {sim}")
		if sim > best_sim:
			best_sim = sim
			best_cat = cat
	print(f"[assign_ticket_to_category] Best category: {best_cat}, Best similarity: {best_sim}, Threshold: {threshold}")
	if best_sim >= threshold:
		return best_cat
	return None

def create_category(ticket, memory, get_embedding, llm_suggest_name):
	"""Create a new category for the ticket. Returns new category name."""
	print(f"[create_category] Creating category for ticket: {ticket}")
	ticket_emb = get_embedding(ticket)
	# Use LLM to suggest a category name and reason
	llm_response = llm_suggest_name([ticket])
	print(f"[create_category] LLM response: {llm_response}")
	# Do not parse here; return raw response for agent.py to handle
	return llm_response, ticket_emb

def merge_categories(cat_a, cat_b, memory):
	"""Merge cat_b into cat_a, update examples and embedding."""
	print(f"[merge_categories] Merging '{cat_b}' into '{cat_a}'")
	if cat_a not in memory['categories'] or cat_b not in memory['categories']:
		print(f"[merge_categories] One or both categories not found.")
		return
	memory['categories'][cat_a]['examples'] += memory['categories'][cat_b]['examples']
	# Average embeddings
	emb_a = np.array(memory['categories'][cat_a]['embedding'])
	emb_b = np.array(memory['categories'][cat_b]['embedding'])
	n_a = len(memory['categories'][cat_a]['examples'])
	n_b = len(memory['categories'][cat_b]['examples'])
	memory['categories'][cat_a]['embedding'] = ((emb_a*n_a + emb_b*n_b)/(n_a+n_b)).tolist()
	del memory['categories'][cat_b]
	print(f"[merge_categories] Merge complete. Remaining categories: {list(memory['categories'].keys())}")

def rename_category(cat, new_name, memory):
	"""Rename a category in memory."""
	print(f"[rename_category] Renaming '{cat}' to '{new_name}'")
	if cat not in memory['categories']:
		print(f"[rename_category] Category '{cat}' not found.")
		return
	memory['categories'][new_name] = memory['categories'].pop(cat)
	print(f"[rename_category] Rename complete. Categories: {list(memory['categories'].keys())}")

def cosine_similarity(a, b):
	a = np.array(a)
	b = np.array(b)
	sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
	# print(f"[cosine_similarity] sim: {sim}")
	return sim
