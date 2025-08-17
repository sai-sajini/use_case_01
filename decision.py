# decision.py: Decision engine logic
from llm import llm_adjust_threshold

def adjust_threshold(memory, current_threshold):
	num_categories = len(memory.get('categories', {}))
	total_tickets = sum(len(cat['examples']) for cat in memory.get('categories', {}).values())
	avg_tickets_per_category = total_tickets / num_categories if num_categories else 0
	print(f"[adjust_threshold] Current: {current_threshold}, Num categories: {num_categories}, Avg tickets/category: {avg_tickets_per_category}")
	resp = llm_adjust_threshold(current_threshold, num_categories, avg_tickets_per_category)
	print(f"[adjust_threshold] LLM response: {resp}")
	if 'INCREASE' in resp.upper():
		print("[adjust_threshold] Increasing threshold.")
		return min(current_threshold + 0.05, 0.99)
	elif 'DECREASE' in resp.upper():
		print("[adjust_threshold] Decreasing threshold.")
		return max(current_threshold - 0.05, 0.01)
	else:
		print("[adjust_threshold] Keeping threshold.")
		return current_threshold

def decide_next_action(state):
	"""
	Decide next action based on state dict.
	Returns one of: 'assign', 'create', 'merge', 'rename', 'adjust_threshold'
	"""
	print(f"[decide_next_action] State: {state}")
	# Simple heuristic: prioritize assignment, then create, then merge, then adjust threshold
	if state.get('unprocessed_tickets'):
		print("[decide_next_action] Action: assign")
		return 'assign'
	if state.get('need_create'):
		print("[decide_next_action] Action: create")
		return 'create'
	if state.get('can_merge'):
		print("[decide_next_action] Action: merge")
		return 'merge'
	if state.get('can_rename'):
		print("[decide_next_action] Action: rename")
		return 'rename'
	print("[decide_next_action] Action: adjust_threshold")
	return 'adjust_threshold'
