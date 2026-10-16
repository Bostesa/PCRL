"""METADATA admission of the admitted state-year (pre-lock; no outcome or eligibility value is read).

Allowed before the science lock (handoff section 3): documentation, headers, file hashes, official raw
record counts and household identifiers. This module downloads the official person zip once, hashes the
zip and the CSV member, reads the CSV header, counts raw records, and reads ONLY the SERIALNO column
(household/GQ identifiers) to count distinct housing-unit and group-quarters serials. It never reads
SEX, OCCP, PINCP, ESR, AGEP or any other value column, and prints no row.
"""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from . import schema

ROOT = Path(__file__).resolve().parent.parent / '.private' / 'acs_replication'


def sha_file(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download(url, dest):
    dest = Path(dest)
    if dest.exists():
        return {'url': url, 'path': str(dest), 'downloaded_now': False}
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (research replication; read-only)'})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, 'xb') as f:
        headers = dict(r.headers.items())
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return {'url': url, 'path': str(dest), 'downloaded_now': True, 'http_headers': {k: v for k, v in headers.items()
                                                                                   if k.lower() in ('last-modified', 'content-length', 'etag', 'content-type')}}


def run(out_path):
    raw = ROOT / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / Path(schema.PERSON_ZIP_URL).name
    dl = download(schema.PERSON_ZIP_URL, zip_path)
    rec = {'schema': 'ACSR1-metadata-admission-v1', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
           'kind': 'METADATA_ADMISSION (documentation, headers, hashes, raw counts, household identifiers only)',
           'download': dl, 'zip_sha256': sha_file(zip_path), 'zip_bytes': zip_path.stat().st_size}
    with zipfile.ZipFile(zip_path) as z:
        rec['zip_members'] = [{'name': i.filename, 'bytes': i.file_size, 'crc': i.CRC} for i in z.infolist()]
        names = [i.filename for i in z.infolist()]
        if schema.PERSON_CSV not in names:
            raise ValueError(f'expected member {schema.PERSON_CSV} absent: {names}')
        h = hashlib.sha256(); n_lines = 0; header = None
        serial_idx = None; hu, gq, other = set(), set(), 0
        with z.open(schema.PERSON_CSV) as fb:
            text = io.TextIOWrapper(fb, encoding='latin-1', newline='')
            reader = csv.reader(text)
            for i, row in enumerate(reader):
                if i == 0:
                    header = row; serial_idx = header.index('SERIALNO'); continue
                n_lines += 1
                s = row[serial_idx]          # identifier only; no other column is touched
                if s[4:6] == 'HU':
                    hu.add(s)
                elif s[4:6] == 'GQ':
                    gq.add(s)
                else:
                    other += 1
        with z.open(schema.PERSON_CSV) as fb:
            for chunk in iter(lambda: fb.read(1 << 20), b''):
                h.update(chunk)
    rec['csv_member_sha256'] = h.hexdigest()
    rec['header'] = header
    rec['n_columns'] = len(header)
    rec['raw_person_records'] = n_lines
    rec['distinct_housing_unit_serials'] = len(hu)
    rec['distinct_group_quarters_serials'] = len(gq)
    rec['serials_with_unexpected_prefix'] = other
    needed = ['RT', 'SERIALNO', 'SPORDER', 'ST', 'ADJINC', 'PWGTP', 'AGEP', 'COW', 'SCHL', 'MAR', 'POBP', 'WKHP', 'ESR', 'OCCP', 'SEX', 'PINCP', 'RELSHIPP']
    rec['required_columns_present'] = {c: c in header for c in needed}
    rec['values_read'] = ['SERIALNO (identifier) only']
    rec['outcome_or_eligibility_values_read'] = False
    docs = ROOT / 'docs'
    rec['documentation'] = {p.name: {'sha256': sha_file(p), 'bytes': p.stat().st_size} for p in sorted(docs.glob('*'))
                            if p.suffix in ('.csv', '.txt', '.xls', '.xlsx', '.pdf')}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'x') as f:
        json.dump(rec, f, indent=2, sort_keys=True)
    return rec


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    r = run(ap.parse_args().out)
    print(json.dumps({k: r[k] for k in ('zip_sha256', 'csv_member_sha256', 'raw_person_records', 'distinct_housing_unit_serials',
                                        'distinct_group_quarters_serials', 'serials_with_unexpected_prefix', 'n_columns')}))
    print('missing required columns:', [c for c, ok in r['required_columns_present'].items() if not ok])
