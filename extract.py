with open('archive/deprecated_ml_baselines/phase_d_training.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

out_lines = []
for line in lines:
    out_lines.append(line)
    if line.startswith('def find_optimal_threshold'):
        # keep going until the end of the function
        pass
    if line.startswith('# ─── Classical ML Pipeline'):
        # pop the last line which was the header
        out_lines.pop()
        break

# Also strip out LogisticRegression from the imports in out_lines
final_lines = []
for line in out_lines:
    if 'LogisticRegression' in line:
        continue
    final_lines.append(line)

with open('src/dataset_builder.py', 'w', encoding='utf-8') as f:
    f.writelines(final_lines)
print("Extraction complete.")
