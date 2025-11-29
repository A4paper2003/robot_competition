import cv2
import numpy as np
import Adafruit_PCA9685
import RPi.GPIO as GPIO
import time

# Initialize the servo motor
pwm = Adafruit_PCA9685.PCA9685(0x41)
pwm.set_pwm_freq(50)

# Define the gun GPIO pins
IN1 = 24
IN2 = 23
ENA = 18

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

# Gun control functions
def motor_on():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(ENA, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    print("Gun firing!")

def motor_off():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(ENA, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    print("Gun stopped")
    
# Initialize the servo motor
pwm = Adafruit_PCA9685.PCA9685(0x41)
pwm.set_pwm_freq(50)

# Function to set the servo angle
def set_servo_angle(channel, angle):
    angle = 4096 * ((angle * 11) + 500) / 20000
    pwm.set_pwm(channel, 0, int(angle))

# Initial servo positions
X_P = 90
Y_P = 90
Z_P = 90
set_servo_angle(1, X_P)
set_servo_angle(2, Z_P)
set_servo_angle(3, Y_P)

# Camera settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # Remove auto exposure
cap.set(3, 640)  # Set frame width
cap.set(4, 480)  # Set frame height

# Define the light green color range in HSV
green_lower = np.array([40, 0, 50])
green_upper = np.array([70, 255, 255])

# Control sensitivity for servo response (higher value = slower response)
control_sensitivity = 10

# PID parameters
Kp = 0.1  # Proportional gain
Ki = 0.0  # Integral gain
Kd = 0.0  # Derivative gain

# Variables for PID control
integral_x = 0
integral_y = 0
previous_error_x = 0
previous_error_y = 0

# Variables to track timing and states
start_time = None
has_centered = False
CENTER_THRESHOLD = 10  # Pixel threshold for "centered"

# Helper function to implement PID control
def pid_control(error, integral, previous_error, Kp, Ki, Kd):
    integral += error
    derivative = error - previous_error
    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV and mask for green color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, green_lower, green_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours for potential circular green clusters
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
       
        # Initialize variable to store the center of detected pattern
        pattern_center = None
       
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:  # Only consider significant contours
                hull = cv2.convexHull(c)
                if len(hull) > 5:  # More points indicate a circular shape
                    (x, y), radius = cv2.minEnclosingCircle(hull)
                    if 20 < radius < 50:  # Adjust radius range for your pattern size
                        pattern_center = (int(x), int(y))
                        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
                        cv2.circle(frame, pattern_center, int(radius), (255, 0, 0), 2)
                        break  # Stop if pattern found


        # Move the servos based on the detected pattern center
        if pattern_center:
            cx, cy = pattern_center
            frame_center_x = (frame.shape[1] // 2) - 50                      # negative value cause target to shift to the left in camera image, positive value to the right "target move right, cam center move left"                                                                                                                                                                                                                                                                                                                                                                                                                                                             
            frame_center_y = (frame.shape[0] // 2) - 165                     # negative value cause target to move up in cam frame, positive value cause target to move down

            # Calculate the difference between the pattern center and the image center
            dx = cx - frame_center_x
            dy = cy - frame_center_y

            # Start timing when movement begins, but only if not already centered
            if not has_centered and start_time is None:
                start_time = time.time()
                print("Green object detected, starting timer.")

            # Apply PID control for smoother, non-oscillating servo movement
            pid_output_x, integral_x = pid_control(dx, integral_x, previous_error_x, Kp, Ki, Kd)
            pid_output_y, integral_y = pid_control(dy, integral_y, previous_error_y, Kp, Ki, Kd)

            # Adjust the servo angles using PID output
            X_P -= pid_output_x / control_sensitivity
            Y_P -= pid_output_y / control_sensitivity

            # Limit the angles within bounds
            X_P = max(0, min(175, X_P))
            Y_P = max(0, min(175, Y_P))

            # Set servo angles for pan and tilt
            set_servo_angle(1, X_P)
            set_servo_angle(3, Y_P)

            # Check if the object is centered and trigger gun
            if abs(dx) < CENTER_THRESHOLD and abs(dy) < CENTER_THRESHOLD and not has_centered:
                elapsed_time = time.time() - start_time
                if elapsed_time >= 0.01:  # Ensure timing is above a threshold
                    motor_on()
                    time.sleep(0.1)     #shotting time
                    motor_off()
                    print("Time taken to center: {:.2f} seconds".format(elapsed_time))
                else:
                    print("Ignored timing as it's too fast.")
                

        else:
            # Reset state when the green object disappears
            if has_centered:
                print("Object lost. Ready for next detection.")
            has_centered = False
            start_time = None
            previous_error_x = 0
            previous_error_y = 0

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user")

finally:
    # Cleanup
    motor_off()
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    print("Cleanup complete")
