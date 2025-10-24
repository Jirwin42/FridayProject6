# FridayProject6
Customer Feedback Analysis Bot
==============================

This project is a Python application that uses AI to analyze customer feedback. It reads reviews from a SQLite database, performs sentiment analysis and aspect extraction using the OpenAI API, and then generates a comprehensive report with data visualizations and actionable recommendations. The final analysis is presented in a simple graphical user interface (GUI).

How It Works
------------

The main script (`main.py`) performs the following steps:

1.  **Connects** to the `feedback.db` database to fetch all customer reviews.

2.  **Analyzes** each review using the OpenAI API (`gpt-4o-mini`) to determine:

    -   **Overall Sentiment**: (Positive, Negative, or Neutral).

    -   **Key Aspects**: (e.g., "Price", "Weight", "Battery") mentioned in the review.

3.  **Aggregates** all the analyzed data.

4.  **Generates** a series of data visualizations (as `.png` files) to summarize the findings.

5.  **Produces** a raw data file (`.json`) of the complete analysis.

6.  **Uses AI** one more time to generate a high-level report (`.md`) with key strengths, areas for improvement, and actionable recommendations.

7.  **Launches** a Tkinter GUI dashboard to display the recommendations and all the generated charts in a tabbed view.

Setup and Installation
----------------------

To run this project, you will need Python 3 and the following packages:

1.  **Install Dependencies**:

    ```
    pip install -r requirements.txt

    ```

2.  **Set Up API Key**: You must have an OpenAI API key to run the analysis.

    -   Create a file named `.env` in the same directory.

    -   Add your API key to this file like so:

        ```
        OPENAI_API_KEY=your_actual_key_here

        ```

Usage
-----

Once the setup is complete, simply run the main Python script from your terminal:

```
python main.py

```

The script will process all reviews from `feedback.db`. It will print its progress to the console, save the analysis files (plots, JSON, and markdown) to the directory, and then automatically open the results dashboard window.

File Descriptions
-----------------

Here is a breakdown of every file in this project:

### Core Application

-   **`main.py`**: The main executable script. It orchestrates the entire process, from data fetching and AI analysis to generating reports and launching the GUI.

-   **`feedback.db`**: A SQLite database file. It contains the raw customer feedback in a table named `reviews`.

-   **`requirements.txt`**: Lists the Python libraries needed for the project:

    -   `openai`: For communicating with the OpenAI API.

    -   `python-dotenv`: For loading the API key from the `.env` file.

    -   `matplotlib`: For generating the data visualization plots.

    -   `Pillow`: (PIL) Used to load and display images in the GUI.

### Configuration

-   **`.gitignore`**: A Git configuration file that specifies which files to ignore from version control. It is set to ignore `.env` (to protect your secret API key) and `requirements.txt`.

-   **`README.md`**: (This file) Provides documentation for the project.

### Analysis Output Files

These files are **generated** by `main.py` every time it runs.

-   **`analysis_raw_data.json`**: A JSON file containing the detailed, review-by-review analysis. Includes the review text, its overall sentiment, and a list of all extracted aspects.

-   **`analysis_recommendations.md`**: A markdown file containing the final AI-generated report. It identifies top strengths, key areas for improvement, and provides specific, actionable recommendations.

-   **`analysis_sentiment_distribution.png`**: A pie chart showing the percentage breakdown of Positive, Negative, and Neutral reviews.

-   **`analysis_aspect_frequency.png`**: A horizontal bar chart that displays the top 10 most frequently mentioned aspects (e.g., Price, Comfort, Weight).

-   **`analysis_aspect_sentiment_breakdown.png`**: A stacked horizontal bar chart showing the sentiment (Positive/Negative/Neutral) for each of the top aspects, helping to pinpoint what users like or dislike about specific features.