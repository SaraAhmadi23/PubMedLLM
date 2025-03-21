#!/usr/bin/env python3
"""
Script to clean null bytes from main.py
"""

def clean_null_bytes(file_path):
    print(f"Cleaning null bytes from {file_path}...")
    
    try:
        # Read the file content
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Remove null bytes
        cleaned_content = content.replace(b'\x00', b'')
        
        # Check if anything was cleaned
        if len(content) != len(cleaned_content):
            print(f"Removed {len(content) - len(cleaned_content)} null bytes")
        else:
            print("No null bytes found")
            
        # Write back the cleaned content
        with open(file_path, 'wb') as f:
            f.write(cleaned_content)
            
        print("File cleaned successfully")
        
    except Exception as e:
        print(f"Error cleaning file: {str(e)}")
        return False
        
    return True

if __name__ == "__main__":
    # Clean main.py
    clean_null_bytes("src/main.py") 