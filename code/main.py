from modules.read_file import read_jsonl
from modules.preprocess import preprocess_text


data = read_jsonl('C:/Users/USER/Documents/CodaLab_Bench/GermEval/sustaineval2025_data/development_data.jsonl')


data['context'] = data['context'].apply(lambda x: preprocess_text(x))
data['target'] = data['target'].apply(lambda x: preprocess_text(x))

print(data.head())