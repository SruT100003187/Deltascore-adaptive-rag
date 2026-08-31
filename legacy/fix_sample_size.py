"""
Corrected version. The previous checker had a bug: it searched the
whole file for the text SAMPLE_SIZE = "all", but that exact text also
appears inside an unrelated message later in the file, which fooled it
into saying "already correct" even when it wasn't.

This version only looks at the one line that actually STARTS WITH
"SAMPLE_SIZE =" -- the real setting -- and ignores everywhere else
that phrase might appear.
"""

with open("02_run_baseline.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

found_line_index = None
for i, line in enumerate(lines):
    if line.strip().startswith("SAMPLE_SIZE ="):
        found_line_index = i
        break

if found_line_index is None:
    print("Could not find a line starting with 'SAMPLE_SIZE =' in the file.")
    print("Please paste me the first 15 lines of 02_run_baseline.py.")
else:
    current_line = lines[found_line_index].strip()
    print(f"The real setting line right now is: {current_line}")

    if 'SAMPLE_SIZE = "all"' in current_line or "SAMPLE_SIZE = 'all'" in current_line:
        print('\nCONFIRMED: this line is genuinely already set to "all". No change made.')
    else:
        lines[found_line_index] = 'SAMPLE_SIZE = "all"                # full run\n'
        with open("02_run_baseline.py", "w", encoding="utf-8") as f:
            f.writelines(lines)
        print('\nFIXED: changed this line to SAMPLE_SIZE = "all" and saved the file.')

# Re-read the file fresh from disk and show the real, final result.
with open("02_run_baseline.py", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip().startswith("SAMPLE_SIZE ="):
            print("\nFinal confirmed line in the file right now:")
            print("  " + line.strip())
            break
