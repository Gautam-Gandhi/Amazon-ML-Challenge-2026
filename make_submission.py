"""Assemble the final submission zip for a run.

    python make_submission.py --run exp01 --team <team_name>

Layout (as required by the challenge):
  <team>_submission.zip
    output/matching_results.tsv, output/candidate_pairs.tsv      (from runs/<run>/output)
    code/business_entity_resolution/src/*.py                    (pipeline source for that run)
    code/business_entity_resolution/README.md, requirements.txt (from submission_kit/<run>/)
    Documentation_template.md                                   (from submission_kit/<run>/)
"""
import os
import zipfile
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_FILES = {
    "exp01": ["er_common.py", "prep_v1.py", "exp01_block.py", "exp01_match.py"],
    "exp02": ["er_common.py", "prep_v1.py", "exp01_block.py", "exp01_match.py", "exp02_stack.py"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp01")
    ap.add_argument("--team", default="team")
    args = ap.parse_args()
    kit = os.path.join(ROOT, "submission_kit", args.run)
    out_dir = os.path.join(ROOT, "runs", args.run, "output")
    os.makedirs(os.path.join(ROOT, "submission"), exist_ok=True)
    zpath = os.path.join(ROOT, "submission", f"{args.team}_submission_{args.run}.zip")
    base = "code/business_entity_resolution"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in ["matching_results.tsv", "candidate_pairs.tsv"]:
            z.write(os.path.join(out_dir, f), f"output/{f}")
        for f in SRC_FILES[args.run]:
            z.write(os.path.join(ROOT, f), f"{base}/src/{f}")
        z.write(os.path.join(kit, "run_pipeline.py"), f"{base}/src/run_pipeline.py")
        z.write(os.path.join(kit, "README.md"), f"{base}/README.md")
        z.write(os.path.join(kit, "requirements.txt"), f"{base}/requirements.txt")
        z.write(os.path.join(kit, "Documentation_template.md"), "Documentation_template.md")
    with zipfile.ZipFile(zpath) as z:
        for i in z.infolist():
            print(f"{i.file_size / 1e6:10.2f} MB -> {i.compress_size / 1e6:8.2f} MB  {i.filename}")
    print("wrote", zpath, f"({os.path.getsize(zpath) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
