import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/home/tuq33671/.cache/huggingface/hub/models--meta-llama--Meta-Llama-Guard-2-8B/snapshots/7d257f3c1a0ec6ed99b2cb715027149dfb9784ef")
model = AutoModelForCausalLM.from_pretrained("/home/tuq33671/.cache/huggingface/hub/models--meta-llama--Meta-Llama-Guard-2-8B/snapshots/7d257f3c1a0ec6ed99b2cb715027149dfb9784ef")
import csv
import re

# Define viruses and questions
viruses = ["Alkhurma virus", "Cache Valley virus", "Caliciviruses", "California encephalitis virus", "Chapare virus", "Chikungunya virus", "Crimean-Congo hemorrhagic fever virus", "Dengue virus", "Eastern equine encephalitis virus", "Ebola virus", "Enterovirus D68", "Enterovirus A71", "Guanarito virus", "Hantaviruses", "Heartland virus", "Hendra virus", "Hepatitis A virus", "Influenza viruses", "Japanese encephalitis virus", "Junin virus", "Kyasanur Forest virus", "LaCrosse encephalitis virus", "Langya virus", "Lassa virus", "Lujo virus", "Lymphocytic choriomeningitis virus", "Machupo virus", "Marburg virus", "Mayaro virus", "Nipah virus", "O’nyong-nyong virus", "Omsk hemorrhagic fever virus", "Oropouche virus", "Poliovirus","Powassan", "Punta Toro virus", "Rabies virus", "Rift Valley fever virus", "St. Louis encephalitis virus","Severe acute respiratory syndrome associated coronavirus", "Severe fever with thrombocytopenia syndrome virus", "Tick-borne encephalitis virus", "Variola virus", "Venezuelan equine encephalitis virus", "West Nile virus", "Western equine encephalitis virus", "Yellow fever virus", "Zika virus"]
viruses = ['Alkhurma virus']

questions = [
    "What number of reference genomes are there for this virus?",
    "What number of sequences are there for each pathogen?",
    "What number are full genome sequences for this virus?",
    "What number of papers have been published about this virus?",
    "Is there a comparative evolutionary analysis for this virus?"
]

# Initialize the table with a header row
header = ["Virus"] + [f"Q{idx}" for idx in range(1, len(questions) + 1)]
table_data = [header]

# Collect numerical answers for each virus
for virus in viruses:
    row = [virus]
    for idx, question in enumerate(questions, start=1):
        # Strong prompt asking for number only
        prompt = f"Virus: {virus}\nQ{idx}: {question}\nA{idx}: (Please respond with numbers only. If not available, say 0.)\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            top_p=1.0,
            repetition_penalty=1.1
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_answer = response[len(prompt):].strip()

        # Extract the first number (handles integers and decimals)
        match = re.search(r'\d+(?:,\d+)*(?:\.\d+)?', raw_answer)
        number = match.group(0).replace(',', '') if match else "0"  # Default to 0 if no number

        row.append(number)
    
    table_data.append(row)

print(table_data)

# Save the table as CSV
with open("virus_numeric_answers2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(table_data)

print("✅ Numerical answers saved to 'virus_numeric_answers2.csv'")
