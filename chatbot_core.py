import re

def smalltalk(user_text: str):
    """
    Handles simple greetings and pleasantries using whole-word matching.
    """
    t = user_text.lower().strip()
    
    # --- KEY CHANGE: Using regex for whole-word matching to avoid partial matches like 'hi' in 'this' ---
    if re.fullmatch(r"h(i|ello|ey)\b.*", t):
        return "Hi! How can I help you today?"
    if "thank" in t:
        return "You're welcome! 😊"
    if any(x in t for x in ["bye", "goodbye", "see you"]):
        return "Goodbye! Have a great day!"
    
    return None