import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
def diff(x, y, path=""):
    out = []
    if type(x) != type(y):
        return [(path, "type", str(x)[:80], str(y)[:80])]
    if isinstance(x, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                out.append((path + "/" + k, "missing", k in x, k in y))
            else:
                out += diff(x[k], y[k], path + "/" + k)
    elif isinstance(x, list):
        if len(x) != len(y):
            return [(path, "len", len(x), len(y))]
        for i, (u, v) in enumerate(zip(x, y)):
            out += diff(u, v, f"{path}[{i}]")
    elif x != y and not path.endswith("sha256"):
        out.append((path, "value", str(x)[:80], str(y)[:80]))
    return out
total = 0
digests_equal = 0
for case in sorted(a):
    d = diff(a[case], b.get(case))
    da, db = a[case].get("digest") or a[case].get("digests"), (b.get(case) or {}).get("digest") or (b.get(case) or {}).get("digests")
    digests_equal += da == db
    if d:
        total += 1
        print(case, len(d), d[:6])
print(f"cases={len(a)} differing={total} digest_equal={digests_equal}")
