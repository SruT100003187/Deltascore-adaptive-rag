"""
Run this once before the full baseline run.

It checks 02_run_baseline.py and makes sure SAMPLE_SIZE is set to "all"
instead of the 20-query test value — and tells you clearly which case
applied, so there's no doubt about whether it worked.
"""

with open("02_run_baseline.py", "r", encoding="utf-8") as f:
    content = f.read()

old_line = "SAMPLE_SIZE = 20"
new_line = 'SAMPLE_SIZE = "all"'

if new_line in content:
    print('ALREADY CORRECT: SAMPLE_SIZE is set to "all".')
    print("You can run 02_run_baseline.py now for the full 400-query run.")
elif old_line in content:
    content = content.replace(old_line, new_line)
    with open("02_run_baseline.py", "w", encoding="utf-8") as f:
        f.write(content)
    print('FIXED: SAMPLE_SIZE has been changed to "all" and saved.')
    print("You can run 02_run_baseline.py now for the full 400-query run.")
else:
    print("Could not find the SAMPLE_SIZE line automatically.")
    print("Open 02_run_baseline.py and check the line near the top manually,")
    print('it should read: SAMPLE_SIZE = "all"')
