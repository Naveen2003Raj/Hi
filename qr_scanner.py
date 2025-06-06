import cv2
from pyzbar.pyzbar import decode

def process_qr_image(image_path):
    """
    Reads the image at image_path, scans for QR code, and returns decoded data as string.
    """
    try:
        image = cv2.imread(image_path)
        decoded_objects = decode(image)

        if not decoded_objects:
            return "❌ No QR code detected."

        qr_data = decoded_objects[0].data.decode('utf-8')
        return qr_data

    except Exception as e:
        print(f"❌ Error in QR decoding: {e}")
        return "❌ Error processing QR code."
