import os

def clean_null_bytes(directory):
    """Remove null bytes from all Python files in the directory."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'rb') as f:
                        content = f.read()
                    
                    if b'\x00' in content:
                        print(f'Cleaning null bytes from {path}')
                        with open(path, 'wb') as f:
                            f.write(content.replace(b'\x00', b''))
                        print(f'Successfully cleaned {path}')
                except Exception as e:
                    print(f'Error with {path}: {e}')

if __name__ == '__main__':
    clean_null_bytes('src')
    print('Done cleaning null bytes from Python files') 