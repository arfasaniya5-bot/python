# emotion_detector_perfect.py
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# ----------------- Emotion mapping -----------------
# We use a model fine-tuned for emotion classification and then
# map its labels to your desired set: happy, sad, angry, fear, neutral.
# Many public models use labels like: anger, joy, sadness, fear, love, surprise. [web:36]
# Here we map them to your 5-emotion scheme.

RAW_TO_TARGET = {
    "anger": "angry",
    "joy": "happy",
    "sadness": "sad",
    "fear": "fear",
    "love": "happy",       # map love into happy
    "surprise": "neutral", # treat surprise as neutral/other
}

TARGET_EMOTIONS = ["happy", "sad", "angry", "fear", "neutral"]


def load_emotion_pipeline():
    """
    Load a pretrained emotion model from Hugging Face.
    This uses a commonly-used English emotion classifier. [web:36]
    You can change 'j-hartmann/emotion-english-distilroberta-base' to try other models
    that output emotions such as anger, joy, sadness, fear, love, surprise.
    """
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    clf = pipeline("text-classification", model=model_name, return_all_scores=True)
    return clf


def map_raw_scores_to_target(raw_scores):
    """
    raw_scores: list of dicts like:
       [{'label': 'anger', 'score': 0.8}, {'label': 'joy', 'score': 0.1}, ...]
    Returns:
       dict: target_emotion -> aggregated_score
    """
    target_scores = {t: 0.0 for t in TARGET_EMOTIONS}

    for item in raw_scores:
        raw_label = item["label"].lower()
        score = float(item["score"])
        if raw_label in RAW_TO_TARGET:
            mapped = RAW_TO_TARGET[raw_label]
        else:
            # any unknown raw label is considered neutral
            mapped = "neutral"
        target_scores[mapped] += score

    # Normalize scores to sum to 1.0 (optional but nice)
    total = sum(target_scores.values())
    if total > 0:
        for k in target_scores:
            target_scores[k] /= total

    return target_scores


def predict_emotion(clf, text: str):
    """
    Run the model on a single sentence and return:
      - predicted_emotion: one of happy, sad, angry, fear, neutral
      - scores: dict of emotion -> probability
    """
    if not isinstance(text, str) or not text.strip():
        return "neutral", {t: 0.0 for t in TARGET_EMOTIONS}

    # Model returns list[list[{"label": "...", "score": ...}]]
    raw = clf(text)[0]
    target_scores = map_raw_scores_to_target(raw)

    # Pick best emotion
    best_emotion = max(target_scores.items(), key=lambda x: x[1])[0]
    return best_emotion, target_scores


def print_top_scores(scores, top_n=3):
    """
    Pretty-print top N scores from the emotion probability dict.
    """
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print("Top probabilities:")
    for label, p in sorted_items:
        print(f"  {label}: {p:.3f}")


def interactive_loop(clf):
    print("\nType a sentence and press Enter. Type 'exit' to quit.")
    while True:
        try:
            text = input("Enter a sentence: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if text.lower() == "exit":
            break

        if not text:
            print("Please type a valid sentence.")
            continue

        emotion, scores = predict_emotion(clf, text)
        print(f"Emotion detected: {emotion}")
        print_top_scores(scores)
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        help="Run a quick demo on some fixed sentences",
        action="store_true",
    )
    args = parser.parse_args()

    print("Loading pretrained emotion model (this may take a moment the first time)...")
    clf = load_emotion_pipeline()
    print("Model loaded.\n")

    if args.demo:
        demo_sentences = [
            "I love my life",
            "I am terrified right now",
            "I am so mad",
            "This is depressing",
            "I feel nothing",
            "This is so sad",
            "I am unhappy",
            "My day was terrible",
            "I am not happy at all",
            "I am not safe",
            "I am pissed off",
            "I hate this",
            "I am not excited today",
            "Today is a great day",
            "I feel neutral",
        ]
        print("Demo predictions on sample sentences:\n")
        
        for s in demo_sentences:
            emotion, scores = predict_emotion(clf, s)
            top_emotion_score = scores[emotion]
            print(f"  '{s}' -> {emotion} (confidence: {top_emotion_score:.3f})")
        print()

    interactive_loop(clf)


if __name__ == "__main__":
    main()
