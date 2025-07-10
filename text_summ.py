import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
# Download required NLTK data
print("Downloading necessary NLTK data...")
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
print("NLTK setup complete.\n")

def summarize(text, n=3):
    print("Starting summary computation...")
    stop_words = set(stopwords.words('english'))
    print(f"Loaded {len(stop_words)} stop-words.")

    words = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words]
    print(f"Tokenized and filtered words; {len(words)} total words remain.")

    # Build word frequency
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    print(f"Computed frequency table with {len(freq)} unique words.")

    # Score each sentence
    sentences = sent_tokenize(text)
    print(f"Text split into {len(sentences)} sentences.")

    scores = {}
    for sent in sentences:
        for w in word_tokenize(sent.lower()):
            if w in freq:
                scores[sent] = scores.get(sent, 0) + freq[w]
    print(f"Calculated sentence scores for {len(scores)} sentences.")

    # Select top-n sentences
    best = sorted(scores, key=scores.get, reverse=True)[:n]
    print(f"Selected top {n} sentences for summary.\n")

    return " ".join(best)

def main():
    print("=== Offline Text Summarizer ===")
    print("Paste your text below. Press Enter twice to finish input:\n")

    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    text = "\n".join(lines).strip()

    if not text:
        print(" No text entered. Exiting.")
        return

    summary = summarize(text, n=3)

    print("\n--- Summary ---\n")
    print(summary)
    print("\n--- End of Summary ---")

if __name__ == "__main__":
    main()


