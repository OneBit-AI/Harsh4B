import json
import urllib.request
import time

def chat_request(messages, max_tokens=100):
    url = "http://127.0.0.1:7860/api/chat"
    payload = json.dumps({
        "messages": messages,
        "system": "You are a helpful assistant.",
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "request_id": f"test_{int(time.time()*1000)}"
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    accumulated_text = ""
    stats = {}
    
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if line.startswith("data: "):
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                    if "token" in data:
                        accumulated_text += data["token"]
                    if data.get("done") and "stats" in data:
                        stats = data["stats"]
                except json.JSONDecodeError:
                    pass
    t_wall = time.time() - t0
    return accumulated_text, stats, t_wall

print("--- Test 1: Single Turn Prompt ---")
text, stats, wall = chat_request([{"role": "user", "content": "What is government in simple terms?"}], max_tokens=60)
print(f"Response: {text[:150]}...")
print(f"Stats: {stats} | Wall time: {wall:.2f}s\n")

print("--- Test 2: Extended Multi-Turn Chat (16 Turns History - The 5 TPS Scenario) ---")
# Build a 16-turn dialogue
history = []
topics = [
    ("hi", "Hello! How can I help you today?"),
    ("what is the sun?", "The Sun is the star at the center of the Solar System. It is a nearly perfect ball of hot plasma."),
    ("how far is it?", "The average distance from the Earth to the Sun is about 93 million miles (150 million kilometers)."),
    ("what is photosynthesis?", "Photosynthesis is the process used by plants to convert light energy into chemical energy."),
    ("can humans do it?", "No, humans cannot perform photosynthesis as we lack chloroplasts and chlorophyll."),
    ("what is gravity?", "Gravity is a fundamental interaction which causes mutual attraction between all things that have mass."),
    ("who discovered it?", "Sir Isaac Newton formulated the classical law of universal gravitation in 1687."),
]
for u, a in topics:
    history.append({"role": "user", "content": u})
    history.append({"role": "assistant", "content": a})

# Now add the 15th and 16th messages
history.append({"role": "user", "content": "Can you explain what Palantir company does?"})

text2, stats2, wall2 = chat_request(history, max_tokens=60)
print(f"Response: {text2[:150]}...")
print(f"Stats: {stats2} | Wall time: {wall2:.2f}s\n")
