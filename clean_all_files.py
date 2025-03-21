#!/usr/bin/env python
"""
Clean null bytes from all Python files in the project to fix encoding issues.
"""

import os

def clean_null_bytes(directory):
    """Remove null bytes from all Python files in the given directory recursively."""
    cleaned_files = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'rb') as f:
                        content = f.read()
                    
                    if b'\x00' in content:
                        print(f"Cleaning null bytes from {path}")
                        with open(path, 'wb') as f:
                            f.write(content.replace(b'\x00', b''))
                        cleaned_files += 1
                except Exception as e:
                    print(f"Error with {path}: {e}")
    
    return cleaned_files

if __name__ == '__main__':
    print("Cleaning null bytes from Python files...")
    count = clean_null_bytes('src')
    print(f"Successfully cleaned {count} files")
    print("Done! Try running your command again.") 