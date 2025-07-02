import os
import unicodedata
import re

# List of raga names from top_30 dictionary keys
raga_names = [
    'Rāgamālika', 'Tōḍi', 'Bhairavi', 'Kamās', 'Kalyāṇi', 'Saurāṣtraṁ', 
    'Śankarābharaṇaṁ', 'Kāmavardani/Pantuvarāḷi', 'Behāg', 'Bēgaḍa', 
    'Sindhubhairavi', 'Mōhanaṁ', 'Sāvēri', 'Kāṁbhōji', 'Rītigauḷa', 
    'Kāpi', 'Nāṭa', 'Suraṭi', 'Pūrvīkaḷyāṇi', 'Hamsadhvani', 
    'Madhyamāvati', 'Hindōḷaṁ', 'Karaharapriya', 'Ānandabhairavi', 
    'Mukhāri', 'Kānaḍa', 'Ābhōgi', 'Nāṭakurinji', 'Śrī', 'Sencuruṭṭi'
]

# Parent directory
parent_dir = "./top30raga_data"

# Create parent directory if it doesn't exist
os.makedirs(parent_dir, exist_ok=True)

for raga in raga_names:
    # Convert to lowercase ASCII characters
    # First, normalize Unicode characters (NFKD form) and encode as ASCII (ignoring non-ASCII)
    ascii_raga = unicodedata.normalize('NFKD', raga).encode('ASCII', 'ignore').decode('ASCII')
    
    # Remove any non-alphanumeric characters (keeps only letters and numbers)
    ascii_raga = re.sub(r'[^a-zA-Z0-9]', '', ascii_raga)
    
    # Convert to lowercase
    ascii_raga = ascii_raga.lower()
    
    # Create the directory
    dir_path = os.path.join(parent_dir, ascii_raga)
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")

print("All directories created successfully.")