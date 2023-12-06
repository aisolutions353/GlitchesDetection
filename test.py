from tkinter import *
from tkinter import filedialog, ttk, messagebox
import cv2
import os
from ultralytics import YOLO
import threading
import time

# Global variable to control the stopping of the thread
stop_thread = False

def get_metadata(video_path):
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(video.get(cv2.CAP_PROP_FPS), 2)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return f"Width: {width}, Height: {height}, FPS: {fps}, Frames: {frame_count}"



def process_video():
    global stop_thread

    video_path = video_path_var.get()
    output_path = output_path_var.get()

    txt_path = os.path.join(os.path.dirname(output_path), "detection_results.txt")

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    start_time = time.time()

    #color of the box
    box_color = (173, 250, 28)  # Lime Green  
    
    model = YOLO("models/best_100.pt")
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)

    result = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'H264'), 30, size)

    with open(txt_path, 'w') as txt_file:
        for i in range(total_frames):
            if stop_thread:
                break

            ret, frame = video.read()
            if ret:
                frame_number += 1
                res = model.predict(frame, conf=0.5)
                detections = len(res[0].boxes.data.tolist())
                # frame = res[0].plot()
                for xyxy in res[0].boxes.xyxy.tolist():
                    x1, y1, x2, y2 = list(map(int, xyxy))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (x1, y1 - 40), (x1 + 80, y1), color=box_color, thickness=-1,
                                  lineType=cv2.LINE_AA)
                    cv2.putText(frame, "glitch", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                result.write(frame)

                end_time = time.time()
                elapsed_time = end_time - start_time
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)

                if detections != 0:
                    txt_file.write(f"Frame {frame_number}: Time = {int(hours)}h {int(minutes)}m {int(seconds)}s : Detections = {detections}\n")

                progress_var.set((i + 1) / total_frames * 100)

                cv2.imshow('Preview', frame)
                cv2.waitKey(1)
            else:
                break
              

    stop_thread = False
    video.release()
    result.release()
    cv2.destroyAllWindows()

    if not stop_thread:
        messagebox.showinfo("Success", "Video Processed Successfully")

    analyze_button.config(text="Analyze")
    progress_var.set(0)
    root.update_idletasks()

def process_video_thread():
    metadata = get_metadata(video_path_var.get())
    actual_metadata_label.config(text=metadata)
    threading.Thread(target=process_video).start()

def toggle_process():
    global stop_thread

    if analyze_button.cget("text") == "Analyze":
        stop_thread = False
        analyze_button.config(text="Cancel")
        process_video_thread()
    else:
        stop_thread = True
        analyze_button.config(text="Analyze")
        progress_var.set(0)
        root.update_idletasks()

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    video_path_var.set(file_path)

def browse_output():
    video_file_name = os.path.basename(video_path_var.get())
    output_path = filedialog.asksaveasfilename(defaultextension=".mp4", initialfile=f"{os.path.splitext(video_file_name)[0]}_results.mp4", filetypes=[("MP4 files", "*.mp4")])
    output_path_var.set(output_path)

root = Tk()
root.title("PureFrameV1 BETA")

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=1)
root.grid_rowconfigure(4, weight=1)
root.grid_rowconfigure(5, weight=1)

root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=3)
root.grid_columnconfigure(2, weight=1)

video_path_var = StringVar()
output_path_var = StringVar()
progress_var = DoubleVar()

Label(root, text="Video Path:").grid(row=0, column=0, sticky=W+E, padx=5, pady=5)
Entry(root, textvariable=video_path_var).grid(row=0, column=1, sticky=W+E, padx=5, pady=5)
Button(root, text="Browse", command=browse_file).grid(row=0, column=2, sticky=W+E, padx=5, pady=5)

Label(root, text="Output Path:").grid(row=1, column=0, sticky=W+E, padx=5, pady=5)
Entry(root, textvariable=output_path_var).grid(row=1, column=1, sticky=W+E, padx=5, pady=5)
Button(root, text="Browse", command=browse_output).grid(row=1, column=2, sticky=W+E, padx=5, pady=5)

Label(root, text="Metadata:").grid(row=2, column=0, columnspan=3, sticky=W+E, padx=5, pady=2)
actual_metadata_label = Label(root, text="")
actual_metadata_label.grid(row=3, column=0, columnspan=3, sticky=W+E, padx=5, pady=5)

progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, variable=progress_var, mode="determinate")
progress_bar.grid(row=4, columnspan=3, sticky=W+E, padx=5, pady=5)

analyze_button = Button(root, text="Analyze", command=toggle_process)
analyze_button.grid(row=5, columnspan=3, sticky=W+E, padx=50, pady=5)

root.mainloop()
