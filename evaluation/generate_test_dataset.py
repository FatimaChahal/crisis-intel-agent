import json
import pandas as pd
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


def generate_test_dataset() -> dict:
    """
    Generate a professional evaluation dataset from Gold data.

    Returns:
        Dictionary with wildfire and off-topic test cases.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

    # Load Gold data — take diverse sample
    df = pd.read_csv("data/gold/modis_fires_gold.csv")

    # Select diverse cases
    high = df[df["severity"] == "HIGH"].head(5)
    medium = df[df["severity"] == "MEDIUM"].head(5)
    low = df[df["severity"] == "LOW"].head(5)
    sample = pd.concat([high, medium, low]).reset_index(drop=True)

    wildfire_questions = []

    # Generate questions from real data
    for _, row in sample.iterrows():
        response = llm.invoke(
            f"""Generate 2 natural questions an emergency operator would ask 
about this wildfire event. Return ONLY the questions, one per line.

Event: {row['summary']}"""
        ).content.strip()

        for q in response.split("\n"):
            q = q.strip().strip("123456789.-) ")
            if len(q) > 10:
                wildfire_questions.append(
                    {
                        "question": q,
                        "expected_topic": "wildfire",
                        "ground_truth": row["summary"],
                    }
                )

    # Add manual diverse questions
    manual_wildfire = [
        {
            "question": "What were the largest wildfires in Greece?",
            "expected_topic": "wildfire",
            "ground_truth": "Greece 2023 had 96610 ha burnt in summer.",
        },
        {
            "question": "Quels incendies ont touché la France en été ?",
            "expected_topic": "wildfire",
            "ground_truth": "France summer wildfires mainly affected Bouches-du-Rhône.",
        },
        {
            "question": "Tell me about Portugal wildfires in 2017",
            "expected_topic": "wildfire",
            "ground_truth": "Portugal 2017 had severe wildfires over 60000 ha.",
        },
        {
            "question": "ما هي أكبر حرائق الغابات في أوروبا؟",
            "expected_topic": "wildfire",
            "ground_truth": "Greece 2023 had the largest wildfire with 96610 ha.",
        },
        {
            "question": "Y a-t-il un risque d'incendie aujourd'hui en Gironde ?",
            "expected_topic": "wildfire",
            "ground_truth": "Gironde has HIGH wildfire risk in summer.",
        },
        {
            "question": "Which season has the most wildfires in Europe?",
            "expected_topic": "wildfire",
            "ground_truth": "Summer is the peak wildfire season in Europe.",
        },
        {
            "question": "What vegetation burns most in Mediterranean wildfires?",
            "expected_topic": "wildfire",
            "ground_truth": "Conifer and sclerophyll vegetation dominate Mediterranean wildfires.",
        },
        {
            "question": "Cuáles fueron los mayores incendios en España?",
            "expected_topic": "wildfire",
            "ground_truth": "Spain had major wildfires especially in summer seasons.",
        },
        {
            "question": "How long do wildfires typically last in Turkey?",
            "expected_topic": "wildfire",
            "ground_truth": "Turkey wildfires like Antalya 2021 lasted around 9 days.",
        },
        {
            "question": "What is the average burnt area per wildfire in Europe?",
            "expected_topic": "wildfire",
            "ground_truth": "Average varies by country and season.",
        },
    ]
    wildfire_questions.extend(manual_wildfire)

    # Off-topic questions
    off_topic_questions = [
        {
            "question": "What is the best restaurant in Paris?",
            "expected_topic": "off_topic",
            "ground_truth": "",
        },
        {
            "question": "How do I make chocolate cake?",
            "expected_topic": "off_topic",
            "ground_truth": "",
        },
        {
            "question": "What is the capital of Germany?",
            "expected_topic": "off_topic",
            "ground_truth": "",
        },
        {
            "question": "Tell me about football results",
            "expected_topic": "off_topic",
            "ground_truth": "",
        },
        {
            "question": "What is the stock price of Apple?",
            "expected_topic": "off_topic",
            "ground_truth": "",
        },
        {
            "question": "Who won the last World Cup?",
            "expected_topic": "off_topic",
            "ground_truth": "",
        },
        {
            "question": "How do I learn Python programming?",
            "expected_topic": "off_topic",
            "ground_truth": "",
        },
        {
            "question": "What is the weather forecast for London?",
            "expected_topic": "off_topic",
            "ground_truth": "",
        },
        {
            "question": "Recommend me a good movie",
            "expected_topic": "off_topic",
            "ground_truth": "",
        },
        {
            "question": "What is the population of France?",
            "expected_topic": "off_topic",
            "ground_truth": "",
        },
        {
            "question": "كيف أطبخ الكسكس؟",
            "expected_topic": "off_topic",
            "ground_truth": "",
        },
        {
            "question": "Qual è la migliore pizza a Napoli?",
            "expected_topic": "off_topic",
            "ground_truth": "",
        },
        {
            "question": "Comment apprendre le français ?",
            "expected_topic": "off_topic",
            "ground_truth": "",
        },
        {
            "question": "What are the best stocks to buy?",
            "expected_topic": "off_topic",
            "ground_truth": "",
        },
        {
            "question": "Tell me a joke",
            "expected_topic": "off_topic",
            "ground_truth": "",
        },
    ]

    dataset = {
        "wildfire_questions": wildfire_questions[:30],
        "off_topic_questions": off_topic_questions,
    }

    with open("evaluation/test_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"✅ Dataset generated:")
    print(f"   Wildfire questions : {len(dataset['wildfire_questions'])}")
    print(f"   Off-topic questions: {len(dataset['off_topic_questions'])}")
    print(
        f"   Total              : {len(dataset['wildfire_questions']) + len(dataset['off_topic_questions'])}"
    )
    return dataset


if __name__ == "__main__":
    generate_test_dataset()
