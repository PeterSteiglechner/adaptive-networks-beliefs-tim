# 05: Report script: tables and markdown verdict

**Labels:** ready-for-agent

**What to build:** A thin `# %%` report script at the repo root, in the style of the `analyse_*.py` scripts. It is configured by constants at the top: the two folders, the output folder, top-k, and toggles for ablations and shared-start runs. It loads, compares, and writes to a git-ignored `compare_out/`:
- every table as CSV and markdown;
- a short markdown summary at the top: failing tests per family, the largest effects, and the files missing on one side.

`.gitignore` gains `compare_out/`.

**Blocked by:** #03

**Status:** done

- [x] Running the script on two folders produces all the tables and the markdown summary in `compare_out/`
- [x] The markdown summary lists the failing tests per family, the top effects and the one-sided files
- [x] `compare_out/` is git-ignored
- [x] A smoke test runs the script on the synthetic folders and checks the files exist (like the notebook smoke test)
