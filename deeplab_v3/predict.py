import cv2
import time
import argparse
import numpy as np

from PIL import Image
from deeplab import DeeplabV3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="image", help="image, video, camera")
    parser.add_argument("--input", type=str, default="test.jpg", help="test.jpg, test.mp4")
    opt = parser.parse_args()

    deeplab = DeeplabV3()

    video_save_path = "detect.mp4"
    video_fps = 25.0

    if opt.mode == "image":
        img = opt.input
        image = Image.open(img)
        r_image = deeplab.detect_image(image)
        r_image.show()
        r_image = cv2.cvtColor(np.array(r_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite("detect.jpg", r_image)
    
    if opt.mode == "video":
        capture = cv2.VideoCapture(opt.input)
        if video_save_path !="":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(deeplab.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            # print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)
            
            # ESC
            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()