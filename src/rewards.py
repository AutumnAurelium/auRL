
poem_topics = [
    "dawn in the mountains",
    "urban solitude",
    "first love",
    "grief and healing",
    "climate change",
    "artificial intelligence",
    "ocean depths",
    "childhood memories",
    "social justice",
    "interstellar travel",
    "seasons changing",
    "dreams and nightmares",
    "ancient ruins",
    "identity and belonging",
    "the passage of time",
    "pandemic isolation",
    "forests at night",
    "digital relationships",
    "migration journeys",
    "forgotten languages",
    "parenthood",
    "musical instruments",
    "war and peace",
    "technological dependence",
    "lost civilizations",
    "human resilience",
    "cosmic insignificance",
    "garden growth",
    "cultural heritage",
    "everyday objects",
    "animal consciousness",
    "city architecture",
    "religious faith",
    "scientific discovery",
    "river journeys",
    "mental health struggles",
    "artistic inspiration",
    "desert landscapes",
    "lunar reflections",
    "human connection",
    "environmental destruction",
    "technological utopia",
    "mythological creatures",
    "culinary experiences",
    "political upheaval",
    "aging process",
    "quantum physics",
    "lost friendship",
    "urban wildlife",
    "forgotten histories"
]

def letter_reward(prompts: list[str], completions: list[str]):
    rewards = []
    for completion in completions:
        poem = completion[-1]["content"]
        
        score = 0
        longest_stride = 0
        
        for c in poem:
            if c.isupper():
                longest_stride += 1
                score += longest_stride
            else:
                longest_stride = 0
        
        if len(poem) > 0:
            rewards.append(score / len(poem))
        else:
            rewards.append(0)
        
    return rewards



