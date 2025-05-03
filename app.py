import streamlit as st
import os
import subprocess
import shutil
from fuzzy import load_articles, predict_winner

# Paths
combine_folder = "combine"
article_json_path = os.path.join(combine_folder, "latest_article.json")

# Function to run run_all_models.py
def run_models(article_text):
    # Save the input article to a temp text file
    with open("input_article.txt", "w") as f:
        f.write(article_text)

    # Run the models by calling run_all_models.py
    try:
        subprocess.run(["python", "run_all_models.py"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Error running models:\n{e.stderr}")
        return False
    return True

# Streamlit UI
st.title("Political Bias Prediction 📰")

# Textbox for article input
article_text = st.text_area("Enter your article here:", height=300)

if st.button("Analyze"):
    if article_text.strip() == "":
        st.error("Please enter an article before submitting.")
    else:
        with st.spinner("Running models..."):
            success = run_models(article_text)
            if not success:
                st.stop()

            # Load all articles from the combine folder
            try:
                articles = load_articles(combine_folder)
                
                if articles:
                    st.subheader("🔍 Per-Article Analysis")
                    for i, article in enumerate(articles, 1):
                        sentiment = article['sentiment_analysis']['sorted_average_sentiment'][0]
                        emotion_scores = article['emotion_detection']['sorted_emotions']
                        avg_emotion = round(sum(score for _, score in emotion_scores) / len(emotion_scores), 3)
                        predicted_bias = article['political_bias_prediction']['predicted_bias']

                        st.markdown(f"""
                        **Article {i}**
                        - Top Sentiment: `{sentiment[0]}` ({sentiment[1]:.2f})
                        - Avg. Emotion Score: `{avg_emotion}`
                        - Predicted Political Bias: `{predicted_bias}`
                        """)

                    # Get winner using fuzzy logic
                    winner = predict_winner(articles)
                    st.success(f"🏆 Final Fuzzy Political Bias: **{winner}**")
                else:
                    st.warning("⚠️ No valid article outputs found.")
            except Exception as e:
                st.error(f"Error loading or processing articles: {str(e)}")