import cv2

def list_cameras(max_index=10):
    available_cameras = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def main():
    max_index = 10  # Adjust this value if you expect more than 10 cameras
    cameras = list_cameras(max_index)
    
    if cameras:
        print("Available camera indices:")
        for cam_index in cameras:
            print(f"Camera {cam_index}")
    else:
        print("No cameras found.")

if __name__ == "__main__":
    main()
