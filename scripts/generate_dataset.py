import json
import random
import os

random.seed(42)

names = ["Ali", "Ayşe"]
objects = ["kalem", "silgi"]
colors = ["kırmızı", "mavi"]
locations = ["ev", "okul"]

def generate_questions(fact):
    q1 = f"{fact['name']}in {fact['object']}i ne renk?"
    a1 = fact['color']
    q2 = f"{fact['name']} nerede?"
    a2 = fact['location']
    return [(q1, a1), (q2, a2)]

def create_dataset():
    facts = []
    # Just 5 facts to make learning extremely easy and fast
    for i in range(5):
        facts.append({
            "id": i,
            "name": random.choice(names),
            "object": random.choice(objects),
            "color": random.choice(colors),
            "location": random.choice(locations)
        })

    train_data = []
    for f in facts[:3]:
        fact_text = f"{f['name']}in {f['object']}i {f['color']}dir. {f['name']} {f['location']}dedir."
        for q, a in generate_questions(f):
            train_data.append({"context": fact_text, "question": q, "answer": a, "fact_id": f["id"]})

    test_new_knowledge = []
    for f in facts[3:]:
        fact_text = f"{f['name']}in {f['object']}i {f['color']}dir. {f['name']} {f['location']}dedir."
        for q, a in generate_questions(f):
            test_new_knowledge.append({"context": fact_text, "question": q, "answer": a, "fact_id": f["id"]})

    test_update = []
    f = facts[0]
    new_color = "sarı"
    fact_text_1 = f"{f['name']}in {f['object']}i {f['color']}dir."
    fact_text_2 = f"{f['name']}in {f['object']}i {new_color}dir."
    test_update.append({
        "context": [fact_text_1, fact_text_2],
        "question": f"{f['name']}in {f['object']}i ne renk?",
        "answer": new_color,
        "fact_id": f["id"]
    })

    test_delete = []
    test_unseen_patterns = []
    test_2hop = []

    os.makedirs("data", exist_ok=True)
    with open("data/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open("data/test_new_knowledge.json", "w", encoding="utf-8") as f:
        json.dump(test_new_knowledge, f, ensure_ascii=False, indent=2)
    with open("data/test_unseen_patterns.json", "w", encoding="utf-8") as f:
        json.dump(test_unseen_patterns, f, ensure_ascii=False, indent=2)
    with open("data/test_update.json", "w", encoding="utf-8") as f:
        json.dump(test_update, f, ensure_ascii=False, indent=2)
    with open("data/test_delete.json", "w", encoding="utf-8") as f:
        json.dump(test_delete, f, ensure_ascii=False, indent=2)
    with open("data/test_2hop.json", "w", encoding="utf-8") as f:
        json.dump(test_2hop, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    create_dataset()
