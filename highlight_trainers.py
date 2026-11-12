import re
from typing import Literal

Trainer = Literal['Ansh Lamba', 'Dhaval Patel', 'Piyush Garg', 'Edureka', 'Keerti Purswani', 'Hugging Face', 'COMMAND', 'KGP Talkie', 'Sudarshan Koirala', 'Krish Naik', 'John Savill']
TRAINERS = ('Ansh Lamba', 'Dhaval Patel', 'Piyush Garg', 'Edureka', 'Keerti Purswani', 'Hugging Face', 'COMMAND', 'KGP Talkie', 'Sudarshan Koirala', 'Krish Naik', 'John Savill')
ALIASES = {
    'Ansh Lamba': r'ansh(?:\s+lamba)?|lamba',
    'Dhaval Patel': r'dhaval(?:\s+patel)?|patel|codebasics',
    'Piyush Garg': r'piyush(?:\s+garg)?|(?:mr\.?\s+)?garg',
    'Edureka': r'edu\s*reka',
    'Keerti Purswani': r'(?:keerti|kirti)(?:\s+purswani)?|purswani',
    'Hugging Face': r'hugging\s+face',
    'COMMAND': r'command\s+(?:conference|channel)',
    'KGP Talkie': r'kgp\s+talkie|laxmi\s+kant',
    'Sudarshan Koirala': r'koirala|sudarshan(?:\s+koirala)?',
    'Krish Naik': r'(?:krish|krishna)\s+(?:naik|nayak|naick)|naik',
    'John Savill': r'(?:john\s+)?(?:savill|saville|seville)',
}

def named_trainers(question):
    return [name for name in TRAINERS if re.search(r'\b(?:'+ALIASES[name]+r')\b',question,re.I)]

def replace_trainer_names(question, trainer):
    pattern = r'\b(?:'+'|'.join(ALIASES.values())+r')\b'
    return re.sub(pattern,lambda _:trainer,question,flags=re.I)
