import streamlit as st
import nltk

# download bộ dữ liệu về sentiment, dùng cache tránh tải nhiều lần
@st.cache_resource
def load_sentiment_analyzer():
    nltk.download('vader_lexicon')
    from nltk.sentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

sid = load_sentiment_analyzer()

# Streamlit App
def main():
    st.title('Sentiment Analysis with NLTK')
    st.title('NLTK Library')
    st.markdown("Analyze the sentiment of your sentence using **VADER** from the NLTK library.")

    sentence = st.text_input("Enter a sentence to analyze: ")

    # chấm điểm positive và negative
    if sentence:
        sentiment_scores = sid.polarity_scores(sentence)

        # Display all scores
        st.subheader("Sentiment Scores")
        st.write(sentiment_scores)

        # Bar Chart
        st.bar_chart({k: v for k, v in sentiment_scores.items() if k != 'compund'})

        # Sentiment Analysis với màu tùy chỉnh
        compound = sentiment_scores['compound']
        if compound >= 0.05:
            sentiment = "Positive"
            color = "lightgreen"
        elif compound <= -0.05:
            sentiment = "Negative"
            color = "salmon"
        else:
            sentiment = "Neutral"
            color = "lightgray"

        # Hiển thị bằng HTML có đổi màu
        st.markdown(
            f"""
            <div style="background-color:{color}; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                <h3>Sentiment: {sentiment}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
if __name__ == '__main__':
     main()