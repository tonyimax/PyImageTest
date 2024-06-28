# import cv2
# import whisper
# m=whisper.load_model('base')
# r=m.transcribe('test.m4a')
# print(r['text'])
#
# image_origin = cv2.imread('123.png') #原始图像
# image_gray = cv2.imread('123.png',cv2.IMREAD_GRAYSCALE) #灰度图像
# image_edges = cv2.Canny(image_gray, threshold1=100, threshold2=200) #边缘检测图像
#
# cv2.imshow('原始图像',image_origin)
# cv2.imshow('灰度图像', image_gray)
# cv2.imshow('边缘检测图像', image_edges)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# from ultralytics import YOLO
# m=YOLO('yolov8n.pt')
# r=m.train(data='coco8.yaml',epochs=50)
# v=m.val()
# img=m("https://ultralytics.com/images/bus.jpg")
# exp_onnx=m.exprot(format="onnx")
#

# from ultralytics import YOLO
# model = YOLO("yolov8n.pt")
# model.export(format="onnx", dynamic=True)


from ultralytics import Explorer
exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
exp.create_embeddings_table()

similar = exp.get_similar(img="https://ultralytics.com/images/bus.jpg", limit=10)
print(similar.head())

similar = exp.get_similar(
    img=["https://ultralytics.com/images/bus.jpg", "https://ultralytics.com/images/bus.jpg"], limit=10
)
print(similar.head())