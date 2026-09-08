import json, urllib.request, time

def chat(prompt, max_tokens=120):
    url = "http://127.0.0.1:7860/api/chat"
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "system": "You are a helpful assistant.",
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "request_id": f"test_{int(time.time()*1000)}"
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    text = ""
    stats = {}
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if line.startswith("data: "):
                try:
                    d = json.loads(line[6:])
                    if "token" in d: text += d["token"]
                    if d.get("done") and "stats" in d: stats = d["stats"]
                except Exception:
                    pass
    return text, stats

print("Testing Python Binary Search prompt...")
text, stats = chat("Write an optimized binary search in Python with docstrings and comments.", max_tokens=100)
print(f"Generated text:\n{text}\n")
print(f"Stats: {stats}")
