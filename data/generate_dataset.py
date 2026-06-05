"""
data/generate_dataset.py
------------------------
Simulates a realistic Reddit-style mental health dataset.
In production, replace with actual datasets:
  - SMHD (Self-reported Mental Health Diagnoses) from CLPsych
  - Reddit Mental Health Dataset (Kaggle)
  - University of Maryland CLPsych datasets

Labels: 0 = Normal, 1 = Depression, 2 = Anxiety
"""

import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

# ── Templates per class ──────────────────────────────────────────────
TEMPLATES = {
    0: [  # Normal
        "Had a great day today, went hiking with friends and it was wonderful.",
        "Just finished cooking dinner, tried a new recipe and it turned out amazing!",
        "Excited about the weekend plans, going to the cinema with my family.",
        "Work was productive today, hit all my targets and feeling good.",
        "Watched a funny movie last night, laughed so much my sides hurt.",
        "Finally got my garden sorted, planted tomatoes and sunflowers.",
        "Caught up with an old friend over coffee, great conversation.",
        "Finished reading that novel, the ending was unexpected but satisfying.",
        "Morning run felt amazing today, hit a new personal best!",
        "Celebrated a colleague's promotion at the office, it was a fun event.",
        "Spending the evening learning guitar, slowly getting better each day.",
        "Visited my parents this weekend, had a wholesome family dinner.",
        "Felt very grateful today for all the good things in my life.",
        "Just got promoted at work! All the hard work is finally paying off.",
        "Started a new hobby — watercolor painting. It's so relaxing.",
    ],
    1: [  # Depression
        "I can't seem to get out of bed anymore. What's the point of anything.",
        "Everything feels heavy and grey. I don't remember the last time I smiled.",
        "I've been sleeping 14 hours a day and still feel exhausted. Something is wrong.",
        "Lost interest in things I used to love. Gaming, music — nothing feels good.",
        "I feel like a burden to everyone around me. They'd be better off without me.",
        "Crying for no reason again today. I don't even know why I'm so sad.",
        "Skipped work again. Can't bring myself to face anyone. Isolation feels safer.",
        "The emptiness inside me is unbearable. I feel hollow all the time.",
        "I haven't showered in 4 days. Basic tasks feel impossible right now.",
        "Feeling worthless and useless. I keep failing at everything I try.",
        "I've been isolating myself from friends and family. It's easier that way.",
        "No motivation to eat or cook. Just surviving on crackers and water.",
        "My mind keeps replaying all my failures. I can't escape the dark thoughts.",
        "I feel disconnected from everything, like watching my own life from outside.",
        "Crying myself to sleep every night. I don't see a way out of this darkness.",
        "I keep thinking about disappearing. Not death, just... not existing anymore.",
        "Even small decisions feel overwhelming. I'm paralysed by everything.",
        "I've been cancelling all my plans. I can't pretend to be okay anymore.",
    ],
    2: [  # Anxiety
        "My heart is racing again for no reason. I can't stop the panic attacks.",
        "Lying awake at 3am, mind spinning with worst-case scenarios. Can't stop it.",
        "Cancelled plans again because the thought of leaving the house was terrifying.",
        "Constantly checking if I locked the door, turned off the stove. It's exhausting.",
        "Shaking during my presentation today. Everyone must have noticed. So embarrassing.",
        "Racing thoughts keeping me awake. What if I fail? What if everything falls apart?",
        "Feeling suffocated by worry. Every little thing feels like a catastrophe.",
        "Had to leave the supermarket halfway through. Too many people, felt overwhelmed.",
        "My chest tightens whenever my phone rings. Constant dread for no clear reason.",
        "I over-analyse every text message I send. Did I say something wrong?",
        "Persistent feeling of impending doom. I'm waiting for something bad to happen.",
        "Stomach in knots all day at work. Overthinking every interaction with my boss.",
        "Can't concentrate. My mind jumps from one worry to the next nonstop.",
        "Spent hours rehearsing a 5-minute conversation in my head. Still messed it up.",
        "Hyperventilating on the drive to work today. Almost had to pull over.",
        "Avoiding social gatherings — the idea of making small talk fills me with dread.",
        "I feel like I'm always on edge, waiting for the next disaster to strike.",
        "Obsessing over health symptoms online again. Convinced something is terribly wrong.",
    ]
}

NOISE_ADDITIONS = [
    " honestly idk anymore",
    " lol not sure",
    " just my thoughts",
    " maybe it's just me",
    " ugh",
    "",
    " smh",
    " fr fr",
    "",
    " ngl",
]

TYPO_MAP = {
    "the": "teh", "and": "adn", "I": "i",
    "feel": "fel", "really": "rly", "because": "bc",
    "something": "smth", "everyone": "evry1",
}


def inject_noise(text: str, noise_prob: float = 0.3) -> str:
    """Randomly add social-media-style noise to text."""
    if random.random() < noise_prob:
        text += random.choice(NOISE_ADDITIONS)
    # Randomly lowercase some sentences
    if random.random() < 0.4:
        text = text.lower()
    # Inject occasional typos
    if random.random() < 0.2:
        words = text.split()
        words = [TYPO_MAP.get(w, w) for w in words]
        text = " ".join(words)
    # Add URLs occasionally (noise)
    if random.random() < 0.1:
        text += " http://t.co/xyz"
    # Add emojis occasionally
    emojis = ["😔", "😢", "😰", "😊", "❤️", "💔", "😭", "🥺", "😤"]
    if random.random() < 0.15:
        text += " " + random.choice(emojis)
    return text


def generate_dataset(n_samples: int = 3000) -> pd.DataFrame:
    """Generate a balanced-ish simulated mental health text dataset."""
    records = []
    label_names = {0: "normal", 1: "depression", 2: "anxiety"}

    # Slightly imbalanced to reflect reality
    class_counts = {0: int(n_samples * 0.40), 1: int(n_samples * 0.35), 2: int(n_samples * 0.25)}

    for label, count in class_counts.items():
        templates = TEMPLATES[label]
        for i in range(count):
            base = random.choice(templates)
            # Augment: combine two sentences occasionally
            if random.random() < 0.3:
                extra = random.choice(templates)
                base = base + " " + extra
            noisy = inject_noise(base)
            records.append({
                "text": noisy,
                "label": label,
                "label_name": label_names[label],
                "source": "simulated_reddit"
            })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_dataset(n_samples=3000)
    df.to_csv("data/raw/mental_health_dataset.csv", index=False)
    print(f"Dataset saved: {len(df)} rows")
    print(df["label_name"].value_counts())
