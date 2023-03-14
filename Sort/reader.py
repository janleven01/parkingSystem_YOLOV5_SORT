from multiprocessing import Process, Lock
from multiprocessing.shared_memory import SharedMemory
from time import sleep
import cv2
import numpy as np

def read_and_process_frames(process_frames):
 
    #get the first frame to calculate size
    cap = cv2.VideoCapture('videos/test4.mp4')
    success, frame = cap.read()
    if not success:
        raise Exception("error reading from video")
    
    #create the shared memory for the frame buffer
    frame_buffer_shm = SharedMemory(name="frame_buffer", create=True, size=frame.nbytes)
    frame_buffer = np.ndarray(frame.shape, buffer=frame_buffer_shm.buf, dtype=frame.dtype)
    
    frame_lock = Lock()
    exit_flag = Lock()
    exit_flag.acquire() #start in a locked state. When the reader process successfully acquires; that's the exit signal
    
    processing_process = Process(target=process_frames, args=(frame_buffer_shm, 
                                                              frame.shape, 
                                                              frame.dtype, 
                                                              frame_lock, 
                                                              exit_flag))
    processing_process.start()
    
    try: #use keyboardinterrupt to quit
        while True:
            with frame_lock:
                cap.read(frame_buffer) #read data into frame buffer
            sleep(1/30) #limit framerate-ish for video file (hitting actual framerate is more complicated than 1 line)
    except KeyboardInterrupt:
        print("exiting")
    
    exit_flag.release() #signal the child process to exit
    processing_process.join() #wait for child to exit
    
    #cleanup
    cap.release()
    frame_buffer_shm.close()
    frame_buffer_shm.unlink()