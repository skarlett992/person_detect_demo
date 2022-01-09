from model_zoo.model_api import models

def get_model(model_adapter, args):
    if args.architecture_type == 'ssd':
        return models.SSD(model_adapter, labels=args.labels, resize_type=args.resize_type,
                          threshold=args.prob_threshold)
    elif args.architecture_type == 'ctpn':
        return models.CTPN(model_adapter, input_size=args.input_size, threshold=args.prob_threshold)
    elif args.architecture_type == 'yolo':
        return models.YOLO(model_adapter, labels=args.labels, resize_type=args.resize_type,
                           threshold=args.prob_threshold)
    elif args.architecture_type == 'yolov3-onnx':
        return models.YoloV3ONNX(model_adapter, labels=args.labels, resize_type=args.resize_type,
                                 threshold=args.prob_threshold)
    elif args.architecture_type == 'yolov4':
        return models.YoloV4(model_adapter, labels=args.labels,
                             threshold=args.prob_threshold, resize_type=args.resize_type,
                             anchors=args.anchors, masks=args.masks)
    elif args.architecture_type == 'yolof':
        return models.YOLOF(model_adapter, labels=args.labels, resize_type=args.resize_type,
                            threshold=args.prob_threshold)
    elif args.architecture_type == 'yolox':
        return models.YOLOX(model_adapter, labels=args.labels, threshold=args.prob_threshold)
    elif args.architecture_type == 'faceboxes':
        return models.FaceBoxes(model_adapter, threshold=args.prob_threshold)
    elif args.architecture_type == 'centernet':
        return models.CenterNet(model_adapter, labels=args.labels, threshold=args.prob_threshold)
    elif args.architecture_type == 'retinaface':
        return models.RetinaFace(model_adapter, threshold=args.prob_threshold)
    elif args.architecture_type == 'ultra_lightweight_face_detection':
        return models.UltraLightweightFaceDetection(model_adapter, threshold=args.prob_threshold)
    elif args.architecture_type == 'retinaface-pytorch':
        return models.RetinaFacePyTorch(model_adapter, threshold=args.prob_threshold)
    elif args.architecture_type == 'detr':
        return models.DETR(model_adapter, labels=args.labels, threshold=args.prob_threshold)
    else:
        raise RuntimeError('No model type or invalid model type (-at) provided: {}'.format(args.architecture_type))
