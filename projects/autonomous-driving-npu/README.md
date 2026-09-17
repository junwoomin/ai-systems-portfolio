# Autonomous Driving NPU

## 범위

`YOLO / Segmentation → Multi-task → BEVFormer → TCP / ST-P3`

## 진행 원칙

처음부터 대형 end-to-end 모델 전체를 포팅하지 않습니다.

1. 단일 카메라 perception baseline
2. detection + segmentation multi-task
3. multi-camera feature extraction
4. BEV transformation
5. planning/control head
6. 병목 operator 단위 accelerator port

## 평가 항목

- perception metric: mAP, mIoU
- planning metric: waypoint error
- control 또는 driving score
- end-to-end latency
- peak memory
- accelerator 지원·미지원 operator 목록
