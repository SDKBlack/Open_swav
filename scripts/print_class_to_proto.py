#!/usr/bin/env python3
import sys, os, json
if len(sys.argv) < 2:
    print('Usage: print_class_to_proto.py <mapping_json> [topk]')
    sys.exit(1)
fn = sys.argv[1]
topk = int(sys.argv[2]) if len(sys.argv)>2 else 5
with open(fn,'r',encoding='utf-8') as f:
    data = json.load(f)
mapping = data['mapping']
# aggregate class -> list of (proto, count)
class_map = {}
for m in mapping:
    pid = m['prototype']
    for cls, cnt in m.get('class_counts', {}).items():
        class_map.setdefault(str(cls), []).append((pid, cnt))
# print sorted
for cls in sorted(class_map.keys(), key=lambda x:int(x)):
    lst = sorted(class_map[cls], key=lambda x:-x[1])[:topk]
    s = ', '.join([f"p{p}:{c}" for p,c in lst])
    total = sum(c for _,c in class_map[cls])
    print(f"Class {cls} | total_assigned={total} | top {topk}: {s}")
