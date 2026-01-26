import os
import shutil

os.chdir('C:/Users/SONUR/projects/Novartis')

# Move populate_sample_data.py to utils/
if os.path.exists('populate_sample_data.py'):
    shutil.move('populate_sample_data.py', 'utils/populate_sample_data.py')
    print("✓ Moved populate_sample_data.py to utils/")

# Remove temporary files
for f in ['check_doctors.py', 'merge_changes.ps1']:
    if os.path.exists(f):
        os.remove(f)
        print(f"✓ Removed {f}")

# Remove __pycache__
if os.path.exists('__pycache__'):
    shutil.rmtree('__pycache__')
    print("✓ Removed __pycache__")

print("\n=== Current files ===")
for item in sorted(os.listdir('.')):
    if not item.startswith('.'):
        print(item)

print("\n=== Cleanup complete! ===")
