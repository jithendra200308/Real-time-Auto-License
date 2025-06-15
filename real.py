import cv2
import easyocr
import imutils

# Initialize EasyOCR reader (for English only)
reader = easyocr.Reader(['en'], gpu=True)

# Start video capture (0 = default webcam or CSI camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = imutils.resize(frame, width=640)

    # Perform OCR on the frame
    results = reader.readtext(frame)

    # Loop through detected texts
    for (bbox, text, prob) in results:
        if prob > 0.5:  # Filter low-confidence results
            # Draw bounding box
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            # Display detected text
            cv2.putText(frame, text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show result frame (optional if using display)
    cv2.imshow("ALPR - Jetson Nano", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
