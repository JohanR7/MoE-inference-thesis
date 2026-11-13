import sys
from chakra.schema.protobuf.et_def_pb2 import Node, GlobalMetadata
from chakra.src.third_party.utils.protolib import openFileRd, decodeMessage
fh = openFileRd(sys.argv[1]); meta = GlobalMetadata(); decodeMessage(fh, meta)
nodes = []
while True:
    n = Node()
    if not decodeMessage(fh, n): break
    nodes.append(n)
by_id = {n.id: n for n in nodes}
pf = [n for n in nodes if "prefetch_block" in n.name]
print("nodes", len(nodes), "prefetch loads", len(pf))
for p in pf[:2]:
    parent = [by_id[d].name for d in p.data_deps]
    children = [n.name for n in nodes if p.id in n.data_deps]
    print(f"  {p.name}: after {parent} -> needed by {children}")
