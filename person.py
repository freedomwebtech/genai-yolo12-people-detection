import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import base64
import os
import time
import threading
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ====== Set Your Google Gemini API Key ======
GOOGLE_API_KEY = ""  # Replace with your actual API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

cap = cv2.VideoCapture("person.mp4")

# ====== Initialize Person Direction Tracking ======
person_up = {}
person_down = {}
processed_directions = set()

class PersonDetectionProcessor:
    def __init__(self, yolo_model_path="yolo12s.pt"):
        self.gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        try:
            self.yolo_model = YOLO(yolo_model_path)
            self.names = self.yolo_model.names
        except Exception as e:
            raise RuntimeError(f"❌ Error loading YOLO model: {e}")

        # Define ROI Areas
        self.area = np.array([(421, 380), (445, 399), (662, 318), (643, 299)], np.int32)
        self.area1 = np.array([(387, 350), (412, 372), (640, 293), (618, 274)], np.int32)

        self.processed_track_ids = set()

        # Output Directory and File
        self.current_date = time.strftime("%Y-%m-%d")
        self.output_filename = f"person_data_{self.current_date}.txt"
        self.cropped_images_folder = "cropped_persons"
        os.makedirs(self.cropped_images_folder, exist_ok=True)

        if not os.path.exists(self.output_filename):
            with open(self.output_filename, "w", encoding="utf-8") as file:
                file.write("Timestamp | Track ID | Gender | Dress Color | Clothing Type | Classification | Direction\n")
                file.write("-" * 120 + "\n")

    def analyze_image_with_gemini(self, image_path):
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Analyze this image and extract ONLY the following details:\n\n"
                     "| Gender | Dress Color | Clothing Type |\n"
                     "|--------|-------------|----------------|\n\n"
                     "Important: If the person is Male AND Dress Color is 'Dark Blue/Black' or 'Dark/Black' "
                     "AND Clothing Type is 'Suit', they are a Security Guard. Otherwise, classify them as a Customer."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                     "description": "Detected person"}
                ]
            )

            response = self.gemini_model.invoke([message])
            return response.content.strip()
        except Exception as e:
            print(f"❌ Error invoking Gemini model: {e}")
            return "Error processing image."

    def save_person_info(self, track_id, direction, response_content, timestamp):
        lines = response_content.split("\n")
        valid_rows = []

        for row in lines:
            if "Gender" in row or "---" in row or not row.strip():
                continue
            values = [col.strip() for col in row.split("|")[1:-1]]
            if len(values) == 3:
                gender, dress_color, clothing_type = values

                # === Classify based on AI response ===
                is_security = (
                    gender.lower() == "male" and
                    ("dark" in dress_color.lower() and ("blue" in dress_color.lower() or "black" in dress_color.lower())) and
                    "suit" in clothing_type.lower()
                )
                classification = "Security Guard" if is_security else "Customer"
                valid_rows.append((gender, dress_color, clothing_type, classification))

        if valid_rows:
            with open(self.output_filename, "a", encoding="utf-8") as file:
                for gender, dress_color, clothing_type, classification in valid_rows:
                    file.write(f"{timestamp} | Track ID: {track_id} | {gender} | {dress_color} | {clothing_type} | {classification} | {direction}\n")
            print(f"✅ Data saved for track ID {track_id} - {gender}, {dress_color}, {clothing_type}, {classification}, {direction}.")

    def process_crop_image(self, image, track_id, direction):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        image_filename = os.path.join(self.cropped_images_folder, f"person_{track_id}_{timestamp}.jpg")
        cv2.imwrite(image_filename, image)

        response_content = self.analyze_image_with_gemini(image_filename)
        self.save_person_info(track_id, direction, response_content, timestamp)

    def crop_and_process(self, frame, box, track_id, direction):
        if (track_id, direction) in processed_directions:
            return
        processed_directions.add((track_id, direction))

        x1, y1, x2, y2 = map(int, box)
        cropped_image = frame[y1:y2, x1:x2]

        threading.Thread(target=self.process_crop_image,
                         args=(cropped_image, track_id, direction),
                         daemon=True).start()

    def process_video_frame(self, frame):
        frame = cv2.resize(frame, (1020, 500))
        results = self.yolo_model.track(frame, persist=True)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if self.names[class_id] != "person":
                    continue

                x1, y1, x2, y2 = map(int, box)

                if cv2.pointPolygonTest(self.area, (x2, y2), False) >= 0:
                    person_up[track_id] = (x2, y2)

                if track_id in person_up:
                    if cv2.pointPolygonTest(self.area1, (x2, y2), False) >= 0:
                        direction = "Up"
                        self.draw_info(frame, box, track_id)
                        self.crop_and_process(frame, box, track_id, direction)

                if cv2.pointPolygonTest(self.area1, (x2, y2), False) >= 0:
                    person_down[track_id] = (x2, y2)

                if track_id in person_down:
                    if cv2.pointPolygonTest(self.area, (x2, y2), False) >= 0:
                        direction = "Down"
                        self.draw_info(frame, box, track_id)
                        self.crop_and_process(frame, box, track_id, direction)

        return frame

    @staticmethod
    def draw_info(frame, box, track_id):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cvzone.putTextRect(frame, f"ID: {track_id}", (x2, y2), 1, 1)

    @staticmethod
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            print(f"Mouse Position: ({x}, {y})")

    def start_processing(self):
        cv2.namedWindow("Person Detection")
        cv2.setMouseCallback("Person Detection", self.mouse_callback)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Stream ended or failed.")
                break

            frame_count += 1
            if frame_count % 3 != 0:
                continue

            processed_frame = self.process_video_frame(frame)
            cv2.polylines(processed_frame, [self.area], True, (255, 0, 0), 2)
            cv2.polylines(processed_frame, [self.area1], True, (0, 255, 0), 2)

            cv2.imshow("Person Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("🛑 Exiting...")
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = PersonDetectionProcessor()
    processor.start_processing()
