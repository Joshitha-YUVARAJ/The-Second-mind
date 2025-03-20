import os
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import openai
from dotenv import load_dotenv

# Load API keys from environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OpenAI API key is missing. Set it in the .env file.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Hugging Face Models
generation_model = pipeline("text2text-generation", model="google/t5-small")
reflection_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
ranking_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
evolution_model = pipeline("fill-mask", model="microsoft/deberta-v3-base")
proximity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Memory for past queries
past_interactions = []

# AI Supervisor Function
def ai_assistant(user_input):
    print("\n🔍 User Query:", user_input)
    
    # 1️⃣ Generation: Create a hypothesis or answer
    hypothesis = generation_model(user_input, max_length=50, do_sample=True)[0]["generated_text"]
    print("\n💡 Generated Response:", hypothesis)

    # 2️⃣ Reflection: Check coherence
    labels = ["Valid", "Invalid", "Needs Refinement"]
    reflection = reflection_model(hypothesis, candidate_labels=labels)
    print("\n🔎 Reflection:", reflection["labels"][0], "Score:", reflection["scores"][0])

    # 3️⃣ Ranking: Score the hypothesis
    ranking = ranking_model(hypothesis)[0]
    print("\n📊 Ranking Score:", ranking["label"], "Confidence:", ranking["score"])

    # 4️⃣ Evolution: Refine the answer
    evolved = evolution_model(f"{hypothesis} is [MASK].")
    refined_hypothesis = evolved[0]["sequence"].replace("[MASK]", evolved[0]["token_str"])
    print("\n🔄 Refined Answer:", refined_hypothesis)

    # 5️⃣ Proximity: Recall past interactions
    if past_interactions:
        query_embeddings = proximity_model.encode(past_interactions)
        new_embedding = proximity_model.encode(refined_hypothesis)
        similarities = util.cos_sim(new_embedding, query_embeddings)
        best_match = past_interactions[similarities.argmax()]
        print("\n🔗 Memory Link Found:", best_match)
    else:
        best_match = "No past data."

    # Store new interaction
    past_interactions.append(refined_hypothesis)

    # 6️⃣ Meta-Review: Evaluate the process
    review_prompt = f"Evaluate this AI response process for the query: {user_input}. Generated: {hypothesis}, Refined: {refined_hypothesis}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": review_prompt}]
    )
    print("\n📌 Meta-Review:", response['choices'][0]['message']['content'])

# Run AI Assistant in a loop for continuous conversation
if __name__ == "__main__":
    while True:
        user_query = input("\n🤖 Ask me anything (or type 'exit' to stop): ")
        if user_query.lower() == "exit":
            print("👋 Goodbye!")
            break
        ai_assistant(user_query)
