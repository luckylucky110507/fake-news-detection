#!/usr/bin/env python3
import os
import shutil
import subprocess

os.chdir(r'c:\Users\princ\OneDrive\Desktop\fake news detection')

# Files to remove
files_to_remove = ['packages.txt', 'setup.sh', 'DEPLOYMENT_FIXES.md']

for file in files_to_remove:
    try:
        if os.path.exists(file):
            os.remove(file)
            print(f'✓ Removed: {file}')
    except Exception as e:
        print(f'✗ Error removing {file}: {e}')

# Remove secrets.toml from .streamlit
secrets_file = os.path.join('.streamlit', 'secrets.toml')
try:
    if os.path.exists(secrets_file):
        os.remove(secrets_file)
        print(f'✓ Removed: {secrets_file}')
except Exception as e:
    print(f'✗ Error removing {secrets_file}: {e}')

# Git operations
try:
    subprocess.run(['git', 'add', '-A'], check=True, capture_output=True)
    print('✓ Git add -A')
    
    result = subprocess.run(
        ['git', 'commit', '-m', 'Cleanup: Remove unnecessary files (packages.txt, setup.sh, DEPLOYMENT_FIXES.md) - keep only Streamlit-ready files'],
        check=True,
        capture_output=True,
        text=True
    )
    print('✓ Git commit successful')
    
    result = subprocess.run(['git', 'push', 'origin', 'main'], check=True, capture_output=True, text=True)
    print('✓ Git push successful')
    print('\n✅ Repository cleaned and updated on GitHub!')
except subprocess.CalledProcessError as e:
    print(f'✗ Git error: {e.stderr}')
except Exception as e:
    print(f'✗ Unexpected error: {e}')
