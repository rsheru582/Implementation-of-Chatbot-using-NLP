import spacy
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import pipeline
import numpy as np

# Load spaCy's pre-trained model for text preprocessing
nlp = spacy.load("en_core_web_sm")

# Initialize a response generator (GPT-2 in this case)
response_generator = pipeline("text-generation", model="gpt2")

# Example intents and training phrases
intents = {
    "greeting": ["Hello", "Hi", "Hey", "Good morning", "Good evening"],
    "goodbye": ["Bye", "Goodbye", "See you", "Take care"],
    "weather": ["What is the weather like?", "Tell me the weather", "Is it sunny today?"],
    "thanks": ["Thank you", "Thanks", "Appreciate it"]
}

# Preprocess training phrases
training_sentences = []
training_labels = []

for intent, phrases in intents.items():
    for phrase in phrases:
        training_sentences.append(phrase)
        training_labels.append(intent)

# Vectorize the sentences using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)

# Use Naive Bayes for classification
classifier = MultinomialNB()
classifier.fit(X, training_labels)

# Function to recognize the intent of the user's query
def get_intent(query):
    # Preprocess and predict intent
    query_vector = vectorizer.transform([query])
    intent = classifier.predict(query_vector)[0]
    return intent

# Function to generate a response based on the query and intent
def generate_response(query):
    intent = get_intent(query)
    
    if intent == "greeting":
        return random.choice(["Hello!", "Hi there!", "Hey, how can I assist you today?"])
    
    elif intent == "goodbye":
        return random.choice(["Goodbye!", "See you soon!", "Take care!"])
    
    elif intent == "weather":
        # This is just a placeholder, in a real scenario, you would fetch weather data from an API
        return "The weather is sunny with a chance of rain in the afternoon."
    
    elif intent == "thanks":
        return random.choice(["You're welcome!", "Glad I could help!", "No problem!"])
    
    else:
        # For unknown intents, generate a dynamic response using GPT-2
        response = response_generator(query, max_length=50, num_return_sequences=1)
        return response[0]['generated_text']

# Main chatbot loop
def chat():
    print("Chatbot: Hi! How can I help you today?")
    
    while True:
        query = input("You: ")
        
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        
        response = generate_response(query)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()
