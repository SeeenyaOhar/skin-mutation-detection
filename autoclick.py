import pyautogui
import time

def auto_click(interval=60):
    print("Auto clicker started. Press Ctrl+C to stop.")
    try:
        while True:
            pyautogui.click()  # Click at the current mouse position
            print("Mouse clicked!")
            time.sleep(interval)  # Wait for `interval` seconds
    except KeyboardInterrupt:
        print("\nAuto clicker stopped.")

# Run the function
auto_click()
