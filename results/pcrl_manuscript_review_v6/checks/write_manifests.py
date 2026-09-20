#!/usr/bin/env python3
"""Write papers/pcrl_manuscript_v6/MANIFEST.json and the review SOURCE_MANIFEST.json (hashes only)."""
import json, hashlib, os, glob
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
root = 'papers/pcrl_manuscript_v6'
h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
ver = json.load(open('results/pcrl_manuscript_review_v6/STUDY6_VERIFICATION.json'))
s5 = json.load(open('results/pcrl_manuscript_review_v6/ASSET_HASHES_STUDY6.json'))
files = {os.path.relpath(p, root): h(p) for p in sorted(glob.glob(root + '/**/*', recursive=True))
         if os.path.isfile(p) and not p.endswith(('.aux', '.log', '.bbl', '.blg', '.fls', '.fdb_latexmk', '.out'))
         and not p.endswith('MANIFEST.json')}
json.dump({"manuscript": "pcrl_manuscript_v6", "branch": "research/pcrl-manuscript-integrated-v5",
           "base_commit": "1d18e898a552e920f3ccfc6b22e5f70e2ae734bb",
           "carried_from_v5": "every asset not under generated_study6 is byte-identical to papers/pcrl_manuscript_v5 (see its MANIFEST.json)",
           "generated_study6": {"generator": "results/pcrl_manuscript_review_v6/checks/verify_study6.py",
                                "evidence_commit": ver["evidence_commit"], "read_at": ver["pinned_commit"],
                                "sources_sha256": ver["sources_sha256"], "assets": s5},
           "files_sha256": files, "zero_new_acs_model_fits": True}, open(root + '/MANIFEST.json', 'w'), indent=1)
json.dump({"manuscript_v4": "72a6383320e866cf5dbda1a8465c00c3fe068a3b",
           "study5_final_handoff": "7f961d5c7f6f0562efcb25a27a77bb5221c279a7",
           "study5_evidence": "a56bcc7ff7a122a6411ac3072b11b00b75f4cdad",
           "study5_protocol_lock": "a44b42a1 (as recorded by Study 5)", "study5_panel_freeze": "ad405fe5 (as recorded by Study 5)",
           "study4_evidence": "69e790af36c5ca53203dab17b757a8e3415ee934", "study3": "73903b7f28df68284285f0610a4036beb32b208f",
           "study2": "c37807e4f568ef38e5528fc09c1506083278bf4d", "study1_locked_2017": "349efa454afd907389760fd1f59fd8806a215efd",
           "study6_branch_state": "research/pcrl-utility-extension-aws-v1 at 7f961d5c (no Study 6 commit; tier 0, AWS blocked at v5 publication)",
           "study6_files_sha256": ver["sources_sha256"]},
          open('results/pcrl_manuscript_review_v6/SOURCE_MANIFEST.json', 'w'), indent=1)
print(len(files), "files hashed")
