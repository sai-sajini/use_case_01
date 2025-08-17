# embeddings.py: Embedding model loading and get_embedding()
from sentence_transformers import SentenceTransformer

_model = None

def get_model():
	global _model
	if _model is None:
		print("[get_model] Loading SentenceTransformer model...")
		_model = SentenceTransformer('all-MiniLM-L6-v2')
		print("[get_model] Model loaded.")
	return _model

def get_embedding(text):
	"""Get embedding for a text string as a list of floats."""
	print(f"[get_embedding] Getting embedding for text: {text[:60]}{'...' if len(text) > 60 else ''}")
	model = get_model()
	emb = model.encode([text])[0]
	print(f"[get_embedding] Embedding shape: {getattr(emb, 'shape', type(emb))}")
	return emb.tolist() if hasattr(emb, 'tolist') else list(emb)
