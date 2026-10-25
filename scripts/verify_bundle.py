#!/usr/bin/env python3
"""Verify manifest hashes and scan text plus smoke archive members for secrets."""
import argparse
import gzip
import hashlib
import json
import re
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PATTERNS=[re.compile(rb'\bsk-[A-Za-z0-9_-]{16,}\b'),re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),re.compile(rb'(?i)(?:authorization\s*[:=]\s*[\"\x27]?bearer\s+)[A-Za-z0-9_.-]{15,}')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest',action='store_true')
    args=parser.parse_args()
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.relative_to(ROOT).parts for x in ['.git','.venv','.venv312','.local','outputs','__pycache__','.pytest_cache']))
    findings=[];hashes={}
    def scan(name,data):
        if any(p.search(data) for p in PATTERNS):findings.append(name)
    for p in files:
        rel=str(p.relative_to(ROOT))
        if rel=='manifests/bundle_sha256.json':continue
        b=p.read_bytes();hashes[rel]=hashlib.sha256(b).hexdigest()
        if rel.endswith('.tar.gz'):
            with tarfile.open(p,'r:gz') as t:
                for m in t.getmembers():
                    if m.isfile():scan(rel+':'+m.name,t.extractfile(m).read())
        elif rel.endswith('.gz'):scan(rel,gzip.decompress(b))
        else:scan(rel,b)
    if findings:
        print(json.dumps({'secret_pattern_findings':findings},indent=2));return 1
    manifest=ROOT/'manifests/bundle_sha256.json'
    if args.write_manifest:
        manifest.write_text(json.dumps(hashes,indent=2)+'\n')
    else:
        expected=json.loads(manifest.read_text())
        assert hashes==expected,'Bundle files differ from checksum manifest'
    source=json.loads((ROOT/'manifests/source_snapshot.json').read_text())
    for name,value in source['snapshot_files'].items():
        assert hashlib.sha256((ROOT/'TSM'/name).read_bytes()).hexdigest()==value,name
    print(json.dumps({'files_checked':len(hashes),'bytes':sum((ROOT/n).stat().st_size for n in hashes),'secret_pattern_findings':0,'source_hashes_verified':len(source['snapshot_files']),'note':'Pattern scan only; configuration never includes live credentials.'},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
