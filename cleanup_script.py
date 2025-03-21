#!/usr/bin/env python3
"""
More robust script to clean null bytes from Python files and fix imports
"""
import os
import re

def clean_file(file_path):
    print(f"Cleaning file: {file_path}")
    
    try:
        # Read the file as binary
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Remove null bytes
        cleaned_content = content.replace(b'\x00', b'')
        
        # Check if we removed any null bytes
        null_bytes_removed = len(content) - len(cleaned_content)
        if null_bytes_removed > 0:
            print(f"  - Removed {null_bytes_removed} null bytes")
        
        # Convert to text with strict UTF-8 decoding, replacing invalid chars
        try:
            text_content = cleaned_content.decode('utf-8', errors='replace')
        except UnicodeDecodeError as e:
            print(f"  - Warning: Unicode decode error: {str(e)}")
            text_content = cleaned_content.decode('utf-8', errors='replace')
        
        # Fix relative import to absolute imports for module execution
        if file_path.endswith('main.py'):
            print("  - Fixing imports for module execution")
            
            # Pattern to match relative imports like "from .utils.helpers"
            relative_import_pattern = r'from\s+\.([a-zA-Z0-9_\.]+)\s+import'
            # Pattern to match absolute imports like "from src.utils.helpers"
            absolute_import_pattern = r'from\s+src\.([a-zA-Z0-9_\.]+)\s+import'
            
            if "from src." in text_content:
                # Change src.xxx imports to just xxx when running as module
                text_content = re.sub(absolute_import_pattern, r'from \1 import', text_content)
                print("  - Changed absolute imports to local imports")
            elif "from ." in text_content:
                # Change .xxx imports to just xxx when running as module
                text_content = re.sub(relative_import_pattern, r'from \1 import', text_content)
                print("  - Changed relative imports to local imports")
        
        # Write back as UTF-8
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        print(f"  - File cleaned and saved with UTF-8 encoding")
        return True
        
    except Exception as e:
        print(f"Error cleaning file {file_path}: {str(e)}")
        return False

def clean_directory(directory):
    print(f"Cleaning Python files in {directory}")
    
    cleaned_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                if clean_file(full_path):
                    cleaned_count += 1
    
    print(f"Cleaned {cleaned_count} Python files")
    return cleaned_count

if __name__ == "__main__":
    # Clean main.py first
    clean_file("src/main.py")
    
    # Then clean all other Python files in src directory
    clean_directory("src") 