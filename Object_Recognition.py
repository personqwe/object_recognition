import cv2
from PIL import Image
import torch

def load_yolo_model():
    # YOLOv5 모델 로드
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True) #모델을 저장소에서 load
    return model

def detect_objects(model, webcam_idx=1, width=480, height=640): # 너비와 높이를 640으로 설정
    # 웹캠으로부터 비디오 스트림 가져오기
    cap = cv2.VideoCapture(webcam_idx) #웹캠 띄우기
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width) # 웹캠의 너비 설정
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height) # 웹캠의 높이 설정

    while True:
        # 비디오 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV 프레임을 PIL 이미지로 변환 
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #이미지를 조절하고 회전시킬 수 있도록 변환하는 작업

        # YOLOv5를 사용하여 물체 감지 수행
        results = model(image)

        # 결과 가져오기
        results.print()  # 화면에 결과 출력 (필요에 따라 주석 처리 가능)
        results.save()   # 이미지 파일로 저장 (필요에 따라 주석 처리 가능)

        # 감지된 물체를 프레임에 그리기
        for *xyxy, conf, cls in results.pred[0]: #results.pred 물체정보 리스트 conf 신뢰도 cls 클래스
            label = f'{model.names[int(cls)]} {conf:.2f}' #물체의 이름을 가져옴
            xyxy = [int(x) for x in xyxy] #좌표값을 정수로 변환
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2) #프레임 이미지에 정의된 좌표를 통해 사각형을 그림
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), #프레임 이미지에 지정된 좌표를 추가 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) #cv2.FONT_HERSHEY_SIMPLEX를 사용하며, 크기는 0.5로 지정되고, 색상은 (0, 255, 0)으로 설정

        # 화면에 프레임 표시
        cv2.imshow('YOLOv5 Object Detection', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 비디오 스트림과 창 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # YOLOv5 모델 로드
    model = load_yolo_model()

    # 웹캠으로부터 물체 감지
    detect_objects(model)
