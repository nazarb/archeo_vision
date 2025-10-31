#!/usr/bin/env python3
import sys
print("Python:", sys.executable)
print("Path:", sys.path)

try:
    import io
    print("✓ io imported")
except Exception as e:
    print("✗ io failed:", e)

try:
    from PIL import Image
    print("✓ PIL.Image imported")
except Exception as e:
    print("✗ PIL failed:", e)

try:
    import base64
    image_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    image = Image.open(io.BytesIO(image_bytes))
    print("✓ Full image decode works")
except Exception as e:
    print("✗ Image decode failed:", e)
    import traceback
    traceback.print_exc()
