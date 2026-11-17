"""
inspect_results.py  (SERVER-SIDE)
=================================
통계 보강에 필요한 값들이 어느 결과 파일에 있는지 훑어본다.
어떤 것도 재계산하지 않고, outputs/의 json들을 열어 구조만 보고한다.

실행: python inspect_results.py
"""
import json, os, glob

OUT = "outputs"

def peek(path, maxkeys=40):
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  (열기 실패: {e})"); return
    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"  타입: dict, 최상위 키 {len(keys)}개")
        for k in keys[:maxkeys]:
            v = data[k]
            t = type(v).__name__
            if isinstance(v, (int,float,str)):
                print(f"    {k}: {t} = {v}")
            elif isinstance(v, dict):
                print(f"    {k}: dict, 하위키 = {list(v.keys())[:12]}")
            elif isinstance(v, list):
                sample = v[0] if v else None
                print(f"    {k}: list[{len(v)}], 첫원소 타입 {type(sample).__name__}")
                if isinstance(sample, dict):
                    print(f"       첫원소 키: {list(sample.keys())}")
            else:
                print(f"    {k}: {t}")
    elif isinstance(data, list):
        print(f"  타입: list, 길이 {len(data)}")
        if data and isinstance(data[0], dict):
            print(f"    원소 키: {list(data[0].keys())}")
            print(f"    첫 원소 예시: {json.dumps(data[0], ensure_ascii=False)[:300]}")

print("="*70)
print("outputs/ 의 모든 json 파일")
print("="*70)
jsons = sorted(glob.glob(os.path.join(OUT, "*.json")))
if not jsons:
    print("  (json 파일 없음)")
for p in jsons:
    sz = os.path.getsize(p)
    print(f"\n### {os.path.basename(p)}  ({sz:,} bytes)")
    peek(p)

# context-mass 관련 키를 특별히 탐색
print("\n" + "="*70)
print("context-mass / attention 관련 키 탐색")
print("="*70)
targets = ["context", "mass", "attention", "verbatim", "delta_m", "m_mean", "context_mass"]
for p in jsons:
    try:
        data = json.load(open(p))
    except: 
        continue
    hits = []
    def scan(obj, path=""):
        if isinstance(obj, dict):
            for k,v in obj.items():
                kl = k.lower()
                if any(t in kl for t in targets):
                    hits.append(f"{path}.{k}")
                scan(v, path+"."+k)
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
            scan(obj[0], path+"[0]")
    scan(data)
    if hits:
        print(f"\n{os.path.basename(p)}:")
        for h in hits[:20]:
            print(f"  {h}")

# jsonl 파일도 확인 (probed/eval 세트에 context-mass가 있을 수 있음)
print("\n" + "="*70)
print("jsonl 파일 목록 (probed_/eval_ 등)")
print("="*70)
jsonls = sorted(glob.glob(os.path.join(OUT, "*.jsonl")))[:10]
for p in jsonls:
    print(f"\n### {os.path.basename(p)}")
    try:
        with open(p) as f:
            first = f.readline()
        obj = json.loads(first)
        if isinstance(obj, dict):
            print(f"  키: {list(obj.keys())}")
    except Exception as e:
        print(f"  (읽기 실패: {e})")
