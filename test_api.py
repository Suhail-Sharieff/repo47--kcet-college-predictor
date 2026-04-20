import urllib.request, json

# Test 1: Prediction
r = urllib.request.urlopen('http://localhost:5000/api/predict?rank=5000&category=1G&top_n=10')
data = json.loads(r.read())
print("=== Prediction Test (rank 5000, 1G) ===")
for x in data['results'][:8]:
    print(f"  {x['college_code']} | {x['college_name'][:58]} | cutoff: {x['predicted_cutoff']}")

# Test 2: College list
r2 = urllib.request.urlopen('http://localhost:5000/api/meta')
meta = json.loads(r2.read())
print(f"\n=== Colleges: {len(meta['colleges'])} | Branches: {len(meta['branches'])} ===")

# Find key colleges
keywords = ['visvesvar', 'ramaiah', 'pes university', 'national institute of eng']
print("\nKey deduplicated colleges:")
for c in meta['colleges']:
    nm = c['college_name'].lower()
    if any(k in nm for k in keywords):
        print(f"  {c['college_code']}: {c['college_name']}")

print("\nSample branches (first 15):")
for b in meta['branches'][:15]:
    print(f"  {b}")
