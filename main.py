import os
import sqlite3
import json
from openai import OpenAI
from dotenv import load_dotenv

# --- Setup ---

# Load environment variables (specifically OPENAI_API_KEY) from .env file
load_dotenv()

# --- !! NEW TEST SETTING !! ---
# Set to None to process all reviews, or a number to limit for testing.
LIMIT_REVIEWS = 10 

# Initialize the OpenAI client
# It will automatically look for the OPENAI_API_KEY environment variable
try:
    client = OpenAI()
    # Test if the key is loaded correctly, otherwise, os.environ.get() returns None
    if client.api_key is None:
        raise ValueError("OPENAI_API_KEY not found. Make sure it's in your .env file.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please make sure your .env file is in the same directory and contains:")
    print("OPENAI_API_KEY=your_actual_key_here")
    exit()

# Database file name
DB_FILE = "feedback.db"

# --- Model Definitions ---
# We use gpt-4o-mini as a powerful and cost-effective model.
MODEL_NAME = "gpt-4o-mini"

# --- Function 1: Sentiment Analysis ---

def get_sentiment(review_text: str) -> str:
    """
    Analyzes the sentiment of a single review text.

    Args:
        review_text: The text of the review.

    Returns:
        A string: 'Positive', 'Negative', or 'Neutral'.
    """
    system_prompt = (
        "You are a sentiment analysis expert. Classify the sentiment of the "
        "following review as 'Positive', 'Negative', or 'Neutral'. "
        "Respond with only one of these three words."
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": review_text}
            ],
            temperature=0, # Low temperature for classification tasks
            max_tokens=10
        )
        sentiment = response.choices[0].message.content.strip().replace("\"", "")
        
        # Simple validation to ensure it's one of the expected words
        if sentiment not in ['Positive', 'Negative', 'Neutral']:
            print(f"Warning: Unexpected sentiment '{sentiment}', defaulting to Neutral.")
            return 'Neutral' # Default fallback
            
        return sentiment
        
    except Exception as e:
        print(f"Error in get_sentiment: {e}")
        return "Error"

# --- Function 2: Aspect Extraction (Primary) ---

def extract_aspects(review_text: str) -> list:
    """
    Extracts key aspects (topics, features) from a review.

    Args:
        review_text: The text of the review.

    Returns:
        A list of dictionaries, e.g., 
        [{"aspect": "Battery Life", "quote": "The battery is a joke"}]
        Returns an empty list if no aspects are found or in case of an error.
    """
    system_prompt = (
        "You are an aspect extraction expert. Analyze the following review and "
        "extract key aspects (e.g., 'Battery', 'Comfort', 'Passthrough', 'Price'). "
        "For each aspect, provide a brief quote from the text that supports it. "
        "Respond in JSON format as a list of objects, where each object has "
        "an 'aspect' and 'quote' key. "
        "Example: [{\"aspect\": \"Battery\", \"quote\": \"The battery life is a joke\"}]"
        "\nIf no specific aspects are found, return an empty list []."
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": review_text}
            ],
            # Use JSON mode for reliable, structured output
            response_format={"type": "json_object"},
            temperature=0
        )
        
        result_str = response.choices[0].message.content
        result_data = json.loads(result_str)
        
        # Check if the result is a list (ideal) or a dict containing a list
        if isinstance(result_data, list):
            return result_data
        elif isinstance(result_data, dict) and len(result_data.keys()) == 1:
            # Often the model wraps the list in a key like "aspects"
            key = list(result_data.keys())[0]
            if isinstance(result_data[key], list):
                return result_data[key]
        
        print(f"Unexpected JSON format from extract_aspects: {result_str}")
        return []

    except Exception as e:
        print(f"Error in extract_aspects: {e}")
        return []

# --- !! NEW FUNCTION !! ---
# --- Function 3: Aspect Extraction (Fallback) ---

def get_main_topic_fallback(review_text: str) -> str | None:
    """
    Fallback function to get the single main topic of a review if
    JSON extraction fails.

    Args:
        review_text: The text of the review.

    Returns:
        A string (e.g., "Battery", "Comfort") or None if no topic is found.
    """
    system_prompt = (
        "What is the single main product feature or topic this review is about? "
        "Examples: 'Battery', 'Comfort', 'Price', 'Software', 'Passthrough', 'Games'. "
        "Respond with *only* the topic name. "
        "If the review is too general and has no specific topic, respond with 'General'."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": review_text}
            ],
            temperature=0,
            max_tokens=15 # Enough for 'Mac Virtual Display'
        )
        topic = response.choices[0].message.content.strip().replace("\"", "")
        
        if not topic or len(topic) > 50: # Basic sanity check
            return None
        # Don't log "General" as a specific aspect
        if topic.lower() in ['general', 'general feedback', 'none']:
            return None
            
        return topic
        
    except Exception as e:
        print(f"Error in get_main_topic_fallback: {e}")
        return None

# --- Main Execution (Updated) ---

def analyze_all_reviews():
    """
    Main function to:
    1. Connect to the SQLite database.
    2. Fetch reviews (respecting the LIMIT).
    3. Loop through each review.
    4. Perform sentiment analysis.
    5. Perform primary aspect extraction.
    6. Run fallback extraction if primary fails.
    7. Combine and tag the data.
    8. Print the final results.
    """
    
    print(f"Connecting to database: {DB_FILE}")
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Fetch reviews from the 'reviews' table
        if LIMIT_REVIEWS:
            print(f"--- RUNNING IN TEST MODE: Processing only {LIMIT_REVIEWS} reviews. ---")
            cursor.execute("SELECT id, review_text FROM reviews LIMIT ?", (LIMIT_REVIEWS,))
        else:
            print("--- RUNNING IN PRODUCTION MODE: Processing all reviews. ---")
            cursor.execute("SELECT id, review_text FROM reviews")
            
        all_reviews = cursor.fetchall()
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        print(f"Please make sure '{DB_FILE}' is in the same directory.")
        return
    finally:
        if 'conn' in locals():
            conn.close()
            
    if not all_reviews:
        print("No reviews found in the database.")
        return

    print(f"Found {len(all_reviews)} reviews. Starting analysis...\n")
    
    final_analyzed_data = []

    # Loop through each review (id, review_text)
    for review_id, review_text in all_reviews:
        print(f"--- Analyzing Review ID: {review_id} ---")
        
        # 1. Get overall sentiment
        sentiment = get_sentiment(review_text)
        print(f"Overall Sentiment: {sentiment}")
        
        # 2. Try Primary Aspect Extraction
        aspects = extract_aspects(review_text) # This is a list
        
        # 3. Clean, process, and tag
        processed_aspects = []
        
        # --- !! UPDATED LOGIC !! ---
        if not aspects: # Check if the list is empty
            print("Primary JSON extraction empty. Trying fallback...")
            main_topic = get_main_topic_fallback(review_text)
            
            if main_topic:
                print(f"Fallback found main topic: {main_topic}")
                # Format this topic to match the standard data structure
                processed_aspects = [
                    {
                        "aspect": main_topic,
                        "quote": "N/A (Found by fallback)", # Placeholder quote
                        "tagged_sentiment": sentiment # Use the overall sentiment
                    }
                ]
            else:
                print("Fallback also found no specific topic.")
                # processed_aspects remains an empty list []
                
        else: # Primary extraction SUCCEEDED
            print(f"Extracted Aspects (JSON): {aspects}")
            for aspect_item in aspects:
                # Ensure the aspect_item has the expected keys
                if 'aspect' in aspect_item and 'quote' in aspect_item:
                    tagged_aspect = {
                        "aspect": aspect_item['aspect'],
                        "quote": aspect_item['quote'],
                        "tagged_sentiment": sentiment 
                    }
                    processed_aspects.append(tagged_aspect)
        
        # Create the final data structure for this review
        review_data = {
            "review_id": review_id,
            "overall_sentiment": sentiment,
            "review_text": review_text,
            "tagged_aspects": processed_aspects
        }
        
        final_analyzed_data.append(review_data)
        print("---------------------------------\n")

    # All reviews processed. Print the final combined data.
    print("=== ANALYSIS COMPLETE ===")
    print("Final combined data structure (all reviews):")
    
    # Use json.dumps for a clean, readable printout
    print(json.dumps(final_analyzed_data, indent=2))
    
    # Optionally, save to a file
    output_filename = "analysis_results.json"
    if LIMIT_REVIEWS:
        output_filename = f"analysis_results_TEST_LIMIT_{LIMIT_REVIEWS}.json"
        
    with open(output_filename, "w") as f:
        json.dump(final_analyzed_data, f, indent=2)
    print(f"\nResults also saved to '{output_filename}'")


if __name__ == "__main__":
    analyze_all_reviews()

