import cv2
import time
import board
import busio
import threading
from adafruit_pca9685 import PCA9685
from ultralytics import YOLO
import json

CAMERA_HORIZONTAL = 0.5

# --- CONFIGURATION ---
MODEL_PATH = './model2.onnx'
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CENTER_X = CAMERA_WIDTH * CAMERA_HORIZONTAL

STOP_AT_RED_LINE = False

# --- TUNING ---
TRIAL_COUNT = 0
ROI_VERTICAL_CUTOFF = 0.50
Kp = 0.0008
Kd = 0.0015
BASE_SPEED = 0.3
LANE_WIDTH_PIXELS = 450
START_DELAY = 20
SLOW_SPEED = 0.20
NMS_IOU=0.7
CONF = 0.20
AR_TARGET = 0.5

try:
    with open("values_different.json", "r+") as fi:
        js = json.load(fi)
        TRIAL_COUNT = js.get("TrialCount", 0)
        NMS_IOU = js.get("Iou", 0.7)
        Kp = js.get("Kp", 0.0008)
        Kd = js.get("Kd", 0.0015)
        ROI_VERTICAL_CUTOFF = js.get("CameraCutoff", 0.5)
        BASE_SPEED = js.get("BaseSpeed", 0.27)
        CONF = js.get("Conf", 0.2)
        AR_TARGET = js.get("ArTarget", 0.7)
        HORIZONTAL_OFFSET = js.get("HOffset", 0.5)
        SLOW_SPEED = js.get("SlowSpeed", 0.20)

except:
        print("Loading Error, Ignoring")
# STOP SIGN LOGIC
STOP_DURATION = 2.0
STOP_COOLDOWN = 5.0
BOOST_THRESHOLD_Y = CAMERA_HEIGHT * 0.8

# MOTOR PHYSICS
MIN_MOTOR_POWER = 0.07
MAX_STEER = 0.8

# --- Motor Class ---
class Motor:
    def __init__(self, pca, in1, in2):
        self.pca = pca
        self.in1 = pca.channels[in1]
        self.in2 = pca.channels[in2]

    def set_speed(self, speed):
        if abs(speed) < 0.01:
            pwm = 0
        else:
            abs_s = abs(speed)
            mapped_speed = MIN_MOTOR_POWER + (abs_s * (1.0 - MIN_MOTOR_POWER))
            pwm = int(min(mapped_speed, 1.0) * 65535)

        if speed > 0:
            self.in1.duty_cycle = pwm
            self.in2.duty_cycle = 0
        elif speed < 0:
            self.in1.duty_cycle = 0
            self.in2.duty_cycle = pwm
        else:
            self.stop()
            
    def stop(self):
        self.in1.duty_cycle = 0
        self.in2.duty_cycle = 0

# --- ROBOT LOGIC THREAD ---
def robot_control_loop():
    target_x = CENTER_X
    # 1. Init Hardware
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 100
        left_motors = [Motor(pca, 5, 4), Motor(pca, 7, 6)]
        right_motors = [Motor(pca, 3, 2), Motor(pca, 1, 0)]
    except Exception as e:
        print(f"Hardware Init Error: {e}")
        return

    def set_drive(fwd, steer):
        steer = max(min(steer, MAX_STEER), -MAX_STEER)
        left = fwd + steer
        right = fwd - steer
        max_val = max(abs(left), abs(right))
        if max_val > 1.0:
            left /= max_val
            right /= max_val
        for m in left_motors: m.set_speed(left)
        for m in right_motors: m.set_speed(right)

    def stop_all():
        for m in left_motors + right_motors: m.stop()

    # 2. Load Model
    print("Loading YOLO Model...")
    model = YOLO(MODEL_PATH)
    
    prev_error = 0
    last_stop_time = 0
    
    print("\n--- ROBOT STARTED ---")

    try:
        # Run inference
        #results = model(source=1, stream=True, show=False, conf=0.5, imgsz=640, verbose=False)
        cap = None

        # OpenCV capture so we can flip BEFORE inference
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not cap.isOpened():
            raise RuntimeError("Cannot open camera (VideoCapture(0) failed)")

        # input("--- Robot on Standby, Press Enter to Begin ---")
        delay = START_DELAY
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # Flip BEFORE YOLO inference (fix mirrored webcam)
            frame = cv2.flip(frame, 1)

            # Run inference on the flipped frame
            results = model.predict(source=frame, conf=CONF, iou=NMS_IOU, imgsz=640, verbose=False)
            result = results[0]
            boxes = result.boxes
            
            # --- VISION PROCESSING ---
            best_y_ar = None
            best_w_ar = None
            max_y_area = 0
            max_w_area = 0
            stop_requested = False
            
            current_time = time.time()

            speed = BASE_SPEED
            
            for box in boxes:
                cls = model.names[int(box.cls[0])]
                x, y, w, h = box.xywh[0].tolist()
                
                # Lane Check (Turn Later Logic)
                cutoff_pixel = CAMERA_HEIGHT * ROI_VERTICAL_CUTOFF
                if y < cutoff_pixel: 
                    continue
               
                wls = []

                area = w * h
                if cls == 'yellowline' and area > max_y_area:
                    max_y_area = area
                    best_y_ar = w/h
                    best_y_box = (x, y, w, h)
                elif cls == 'whiteline' and area > max_w_area:
                    max_w_area = area
                    best_w_ar = w/h
                    wls.append((x, y, w, h))


                #if best_y_ar:
                #    y_x, y_y, y_w, y_h = best_y_box
                #    area = 0
                #    max_w_area = 0
                #    best_w_ar = None
                #    for x, y, w, h in wls:
                #        if y > (((y_y - y_w / 2) / (y_x - y_w / 2)) * (x - y_x)) + y_y: continue
                #        area = w*h
                #        if area > max_w_area:
                #            max_w_area = area
                #            best_w_area = w/h

            # --- CONTROL LOGIC ---
            
            if best_w_ar and best_y_ar:
                error = (best_y_ar - best_w_ar)
            elif best_w_ar:
                error = AR_TARGET - best_w_ar
            elif best_y_ar:
                error = best_y_ar - AR_TARGET
            else:
                error = 0
            #error = AR_TARGET - (best_w_ar if best_w_ar else -best_y_ar if best_y_ar else AR_TARGET)
            derivative = error - prev_error
            prev_error = error
            steering = ((error * Kp) + (derivative * Kd))
            
	    # --- DEBUG TEXT OVERLAY ---
            debug_line = (
                f"best_w_ar: {best_w_ar} | " +
		        f"best_y_ar: {best_y_ar} | " +
                f"steering: {steering:.4f}"
            )

            print(debug_line)


            set_drive(BASE_SPEED if best_y_ar else SLOW_SPEED, steering)
    except KeyboardInterrupt:
        print()
        print("Keyboard Interupted")

    except Exception as e:
        print()
        print(f"Robot Loop Error: {e}")
        raise e
        
    finally:
        stop_all()
        print("Robot Loop Ended")

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    while True:
        inp = input((
            f"Value to Change\n"
            f"1) Kp: {Kp}\n" +
            f"2) Kd: {Kd}\n" +
            f"3) Target Aspect Ratio: {AR_TARGET}\n" +
            f"4) Camera Cutoff: {ROI_VERTICAL_CUTOFF}\n" +
            f"5) Base Speed: {BASE_SPEED}\n" +
            f"6) Model Confidence: {CONF}\n" +
            f"7) Horizontal Offset: {CAMERA_HORIZONTAL}\n" +
            f"8) Slow Speed: {SLOW_SPEED}\n" +
            f"9) Intersection Over Union: {NMS_IOU}\n" +
            f"Press Enter to start Trial {TRIAL_COUNT + 1}\n" +
            f">>] "))
        try:
            if inp.lower() == "1" or inp.lower() == "kp": 
                num = float(input("Value: "))
                Kp = num
            if inp.lower() == "2" or inp.lower() == "kd": 
                num = float(input("Value: "))
                Kd = num
            if inp.lower() == "3":
                num = float(input("Value: ")) 
                AR_TARGET = num
            if inp.lower() == "4": 
                num = float(input("Value: "))
                ROI_VERTICAL_CUTOFF = num
            if inp.lower() == "5": 
                num = float(input("Value: "))
                BASE_SPEED = num
            if inp.lower() == "6": 
                num = float(input("Value: "))
                CONF = num
            if inp.lower() == "7":
                num = float(input("Value: "))
                CAMERA_HORIZONTAL = num
                CENTER_X = CAMERA_WIDTH * CAMERA_HORIZONTAL
            if inp.lower() == "8":
                num = float(input("Value: "))
                SLOW_SPEED = num
            if inp.lower() == "9" or inp.lower() == "iou":
                num = float(input("Value: "))
                NMS_IOU = num
            if inp == "":
                robot_control_loop()
                TRIAL_COUNT += 1

            with open("values_different.json", "w+") as fi:
                js = {
                    "TrialCount": TRIAL_COUNT,
                    "Kd": Kd,
                    "Kp": Kp,
                    "CameraCutoff": ROI_VERTICAL_CUTOFF,
                    "BaseSpeed": BASE_SPEED,
                    "Conf": CONF,
                    "ArTarget": AR_TARGET,
                    "HOffset": CAMERA_HORIZONTAL,
                    "SlowSpeed": SLOW_SPEED,
                    "Iou": NMS_IOU
                }
                json.dump(js, fi)
        except ValueError:
            print("Invalid Value")
