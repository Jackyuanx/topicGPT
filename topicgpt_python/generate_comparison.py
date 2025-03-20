import os
import json
import openai

def generate_comparison(model, assign_topics_output, prompt_file, output_file, verbose=True):
    with open(assign_topics_output, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Parse each line as a JSON object
    docs = [json.loads(line) for line in lines]
    
    # Use the first two documents for comparison
    doc1 = docs[0]
    doc2 = docs[1]
    
    # Construct the combined document text including topics
    documents_text = (
        f"Document 1:\nText: {doc1.get('text', '')}\nTopics: {doc1.get('response', '')}\n\n"
        f"Document 2:\nText: {doc2.get('text', '')}\nTopics: {doc2.get('response', '')}"
    )
    
    # Read the prompt template which includes the placeholder {document}
    with open(prompt_file, "r", encoding="utf-8") as pf:
        prompt_template = pf.read()
    
    # Replace the placeholder with the constructed documents text
    final_prompt = prompt_template.replace("{document}", documents_text)
    
    if verbose:
        print("Final prompt sent to OpenAI:")
        print(final_prompt)
    
    # Call the OpenAI API (using ChatCompletion for a conversation-like prompt)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": final_prompt}
        ],
        temperature=0  # Lower temperature for deterministic output
    )
    
    # Extract the generated text from the response
    generated_text = response.choices[0].message["content"].strip()
    
    if verbose:
        print("Generated Comparison:")
        print(generated_text)
    
    # Prepare the output JSON object (here we assign an id of 1)
    output_data = {"id": 1, "comparison": generated_text}
    
    # Write the output as one JSON object (one line) to the output file
    with open(output_file, "w", encoding="utf-8") as outf:
        outf.write(json.dumps(output_data) + "\n")
    
    if verbose:
        print(f"Comparison output saved to {output_file}")