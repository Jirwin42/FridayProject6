import os
import sqlite3
import json
import collections
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt

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

# --- !! NEW FUNCTION !! ---
# --- Function 4: Summarize and Structure Data ---

def summarize_data(final_analyzed_data: list) -> (dict, str):
    """
    Calculates statistics from the analyzed data.

    Args:
        final_analyzed_data: The list of review data dictionaries.

    Returns:
        A tuple containing:
        1. summary_data (dict): Structured data for plotting.
        2. summary_text (str): A human-readable text summary.
    """
    
    # 1. Overall Sentiment Distribution
    sentiment_counts = collections.Counter(
        review['overall_sentiment'] for review in final_analyzed_data
    )
    
    # 2. Aspect Frequency and Sentiment-per-Aspect
    aspect_sentiments = collections.defaultdict(
        lambda: collections.Counter()
    )
    all_aspects = []
    
    for review in final_analyzed_data:
        for aspect in review['tagged_aspects']:
            # Normalize aspect names (e.g., "battery", "Battery" -> "Battery")
            aspect_name = aspect['aspect'].strip().title()
            sentiment = aspect['tagged_sentiment']
            
            aspect_sentiments[aspect_name][sentiment] += 1
            all_aspects.append(aspect_name)
            
    aspect_frequency = collections.Counter(all_aspects)
    
    # --- Prepare Structured Data Output ---
    summary_data = {
        "total_reviews": len(final_analyzed_data),
        "sentiment_distribution": dict(sentiment_counts),
        "aspect_frequency": dict(aspect_frequency.most_common(10)),
        "aspect_sentiments": {
            aspect: dict(sentiments)
            for aspect, sentiments in aspect_sentiments.items()
            if aspect in dict(aspect_frequency.most_common(10)) # Only top 10
        }
    }
    
    # --- Prepare Text Summary Output ---
    summary_text = (
        f"--- Executive Summary of {len(final_analyzed_data)} Reviews ---\n\n"
        f"1. Overall Sentiment Distribution:\n"
    )

    if not final_analyzed_data:
        summary_text += "   - No reviews processed.\n"
    else:
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(final_analyzed_data)) * 100
            summary_text += f"   - {sentiment}: {count} reviews ({percentage:.1f}%)\n"
            
    summary_text += f"\n2. Top {len(summary_data['aspect_frequency'])} Most Mentioned Aspects:\n"
    if not summary_data['aspect_frequency']:
        summary_text += "   - No aspects found.\n"
    else:
        for aspect, count in summary_data['aspect_frequency'].items():
            summary_text += f"   - {aspect}: {count} mentions\n"
        
    summary_text += "\n3. Sentiment Breakdown for Top Aspects:\n"
    if not summary_data['aspect_sentiments']:
        summary_text += "   - No aspect sentiment data available.\n"
    else:
        for aspect, sentiments in summary_data['aspect_sentiments'].items():
            summary_text += f"   - {aspect}:\n"
            for sentiment, count in sentiments.items():
                summary_text += f"     - {sentiment}: {count}\n"
            
    summary_text += "\n--- End of Summary ---\n"
    
    return summary_data, summary_text

# --- !! NEW FUNCTION !! ---
# --- Function 5: Create Visualizations ---

def create_visualizations(summary_data: dict, file_prefix: str):
    """
    Generates and saves plots based on the summary data.

    Args:
        summary_data: The structured data from summarize_data().
        file_prefix: The prefix for saving plot files (e.g., "full" or "test_10").
    """
    
    # Plot 1: Sentiment Distribution (Pie Chart)
    try:
        sentiments = summary_data['sentiment_distribution']
        if not sentiments:
            print("Skipping sentiment chart: No sentiment data.")
        else:
            labels = sentiments.keys()
            sizes = sentiments.values()
            colors = ['#4CAF50' if k == 'Positive' else '#F44336' if k == 'Negative' else '#FFC107' for k in labels] # Green, Red, Amber
            
            plt.figure(figsize=(8, 8))
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=140, textprops={'fontsize': 12, 'fontweight':'bold'})
            plt.title(f'Overall Sentiment Distribution ({summary_data["total_reviews"]} Reviews)', fontsize=16, fontweight='bold')
            plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
            
            filename1 = f"{file_prefix}_sentiment_distribution.png"
            plt.savefig(filename1)
            plt.close()
            print(f"Saved: {filename1}")

    except Exception as e:
        print(f"Error creating sentiment pie chart: {e}")

    # Plot 2: Aspect Frequency (Bar Chart)
    try:
        aspects = summary_data['aspect_frequency']
        if not aspects:
            print("Skipping aspect frequency chart: No aspect data.")
        else:
            labels = list(aspects.keys())
            counts = list(aspects.values())
            
            plt.figure(figsize=(12, 8))
            plt.barh(labels, counts, color='#0288D1') # Blue
            plt.xlabel('Number of Mentions', fontsize=12)
            plt.ylabel('Aspects', fontsize=12)
            plt.title(f'Top {len(labels)} Most Frequent Aspects', fontsize=16, fontweight='bold')
            plt.gca().invert_yaxis() # Display most frequent at the top
            plt.tight_layout()
            
            filename2 = f"{file_prefix}_aspect_frequency.png"
            plt.savefig(filename2)
            plt.close()
            print(f"Saved: {filename2}")

    except Exception as e:
        print(f"Error creating aspect frequency bar chart: {e}")
        
    # Plot 3: Sentiment per Aspect (Grouped Bar Chart)
    # This is more complex, we will create a stacked bar chart
    try:
        aspect_sentiments = summary_data['aspect_sentiments']
        if not aspect_sentiments:
            print("Skipping aspect sentiment chart: No aspect sentiment data.")
        else:
            aspect_names = list(aspect_sentiments.keys())
            
            # Get data for Positive, Negative, Neutral sentiments
            pos_data = [aspect_sentiments[a].get('Positive', 0) for a in aspect_names]
            neg_data = [aspect_sentiments[a].get('Negative', 0) for a in aspect_names]
            neu_data = [aspect_sentiments[a].get('Neutral', 0) for a in aspect_names]
            
            plt.figure(figsize=(14, 10))
            
            # Create the stacked bars
            plt.barh(aspect_names, pos_data, color='#4CAF50', label='Positive')
            plt.barh(aspect_names, neg_data, left=pos_data, color='#F44336', label='Negative')
            
            # Calculate new bottom for Neutral
            bottom_neu = [p + n for p, n in zip(pos_data, neg_data)]
            plt.barh(aspect_names, neu_data, left=bottom_neu, color='#FFC107', label='Neutral')
            
            plt.xlabel('Number of Mentions', fontsize=12)
            plt.ylabel('Aspects', fontsize=12)
            plt.title('Sentiment Breakdown per Aspect', fontsize=16, fontweight='bold')
            plt.legend()
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            filename3 = f"{file_prefix}_aspect_sentiment_breakdown.png"
            plt.savefig(filename3)
            plt.close()
            print(f"Saved: {filename3}")
        
    except Exception as e:
        print(f"Error creating aspect sentiment breakdown: {e}")


# --- !! NEW FUNCTION !! ---
# --- Function 6: Get AI Recommendations ---

def get_recommendations(summary_text: str):
    """
    Uses the AI to generate actionable recommendations based on the summary.
    
    Args:
        summary_text: The human-readable summary from summarize_data().
    """
    print("\n--- Generating AI Recommendations ---")
    
    # Handle case where summary might be empty (e.g., test run with 0 results)
    if "---" not in summary_text or "No reviews processed" in summary_text:
         print("Summary text is empty or contains no data. Skipping AI recommendations.")
         return
         
    system_prompt = (
        "You are a Senior Product Manager at Apple, an expert in product strategy "
        "and user feedback analysis. Your task is to analyze the following customer "
        "feedback summary for the Apple Vision Pro."
        "\n\nBased *only* on the provided summary, please do the following:"
        "\n1.  **Identify Top Strengths:** List the top 2-3 product features that "
        "users love. Briefly explain *why* they are strengths, using data from "
        "the summary."
        "\n2.  **Identify Top Areas for Improvement:** List the top 2-3 product "
        "features that are causing the most negative feedback."
        "\n3.  **Provide Actionable Recommendations:** For each 'Area for Improvement', "
        "propose a specific, actionable recommendation for the engineering or "
        "design teams. Be specific (e.g., 'Initiate a new ergonomics study to "
        "redistribute weight' instead of just 'Fix the weight')."
        "\n\nFormat your response clearly with markdown headings."
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, # Use a powerful model for this
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": summary_text}
            ],
            temperature=0.5, # Allow for some creativity in recommendations
            max_tokens=1000
        )
        recommendations = response.choices[0].message.content
        print(recommendations)
        
    except Exception as e:
        print(f"Error in get_recommendations: {e}")
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
    8. Summarize results.
    9. Create visualizations.
    10. Get AI-driven recommendations.
    """
    
    print(f"Connecting to database: {DB_FILE}")
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Set a file prefix for outputs based on test mode
        file_prefix = "analysis"
        
        # Fetch reviews from the 'reviews' table
        if LIMIT_REVIEWS:
            print(f"--- RUNNING IN TEST MODE: Processing only {LIMIT_REVIEWS} reviews. ---")
            cursor.execute("SELECT id, review_text FROM reviews LIMIT ?", (LIMIT_REVIEWS,))
            file_prefix = f"analysis_TEST_LIMIT_{LIMIT_REVIEWS}"
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

    # --- Main Analysis Loop ---
    for review_id, review_text in all_reviews:
        print(f"--- Analyzing Review ID: {review_id} ---")
        
        # 1. Get overall sentiment
        sentiment = get_sentiment(review_text)
        print(f"Overall Sentiment: {sentiment}")
        
        # 2. Try Primary Aspect Extraction
        aspects = extract_aspects(review_text) # This is a list
        
        # 3. Clean, process, and tag
        processed_aspects = []
        
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
                
        else: # Primary extraction SUCCEEDED
            print(f"Extracted Aspects (JSON): {aspects}")
            for aspect_item in aspects:
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

    # --- Post-Analysis Processing ---
    print("=== ANALYSIS COMPLETE ===")
    
    # 1. Save raw JSON output
    output_filename = f"{file_prefix}_raw_data.json"
    with open(output_filename, "w") as f:
        json.dump(final_analyzed_data, f, indent=2)
    print(f"\nRaw analysis data saved to '{output_filename}'")
    
    # 2. Summarize the data
    summary_data, summary_text = summarize_data(final_analyzed_data)
    print("\n\n" + "="*50)
    print(summary_text)
    print("="*50 + "\n")
    
    # 3. Create Visualizations
    print("--- Generating Visualizations ---")
    create_visualizations(summary_data, file_prefix)
    print("---------------------------------\n")

    # 4. Get AI Recommendations
    get_recommendations(summary_text)
    print("\n=== SCRIPT FINISHED ===")


if __name__ == "__main__":
    analyze_all_reviews()

