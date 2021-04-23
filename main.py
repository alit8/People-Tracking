__title__ = "main.py"
__Version__ = "1.0"
__author__ = "Ali Saberi"
__email__ = "ali.saberi96@gmail.com"
__python_version__ = "3.8.6"

import numpy as np
import cv2
import os
import argparse
import json
from PIL import Image
from matplotlib import pyplot as plt
from imutils.video import FPS

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
import torchvision

from utils import utils, plot

from pytorch_objectdetecttrack.models import Darknet
from pytorch_objectdetecttrack.sort import Sort

from deep_head_pose import hopenet
from deep_head_pose.utils import draw_axis

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

conf_thres = 0.8
nms_thres = 0.4
img_size = 416

face_h_max = 0.3
face_w_min = 0.15
face_w_max = 0.85

mouse_pts = []

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']


def get_gender_model(model_path):
    gender_proto = os.path.join(model_path, 'gender_deploy.prototxt')
    gender_model = os.path.join(model_path, 'gender_net.caffemodel')
    model = cv2.dnn.readNet(gender_model, gender_proto)
    return model


def estimate_gender(blob):
    gender_model.setInput(blob)
    gender_preds = gender_model.forward()
    gender = genderList[gender_preds[0].argmax()]
    return gender


def get_age_model(model_path):
    age_proto = os.path.join(model_path, 'age_deploy.prototxt')
    age_model = os.path.join(model_path, 'age_net.caffemodel')
    model = cv2.dnn.readNet(age_model, age_proto)
    return model


def estimate_age(blob):
    age_model.setInput(blob)
    age_preds = age_model.forward()
    age = ageList[age_preds[0].argmax()]
    return age


def get_head_pose_model(model_path):
    snapshot_path = os.path.join(model_path, 'hopenet.pkl')
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    if not torch.cuda.is_available():
        saved_state_dict = torch.load(snapshot_path, map_location='cpu')
    else:
        saved_state_dict = torch.load(snapshot_path)

    model.load_state_dict(saved_state_dict)
    model.eval()

    return model


def get_detection_model(model_path):
    config_path = os.path.join(model_path, 'yolov3.cfg')
    weights_path = os.path.join(model_path, 'yolov3.weights')
    classes_path = os.path.join(model_path, 'yolov3.cfg')

    # Load model and weights
    model = Darknet(config_path, img_size=img_size)
    model.load_weights(weights_path)

    model.eval()
    classes = utils.load_classes(classes_path)

    return model, classes


def detect_image(img):
    # scale and pad image
    ratio = min(img_size / img.size[0], img_size / img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                         transforms.Pad((max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0),
                                                         max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0)),
                                                        (128, 128, 128)),
                                         transforms.ToTensor(),
                                         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = detection_model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)

    return detections[0]


def estimate_head_pose(img):
    transformations = transforms.Compose([transforms.Scale(224),
                                          transforms.CenterCrop(224), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    img = Image.fromarray(img)

    # Transform
    img = transformations(img)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = Variable(img)

    if torch.cuda.is_available():
        img.cuda()

    with torch.no_grad():
        yaw, pitch, roll = head_pose_model(img)

        yaw_predicted = F.softmax(yaw)
        pitch_predicted = F.softmax(pitch)
        roll_predicted = F.softmax(roll)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

    return yaw_predicted, pitch_predicted, roll_predicted


def get_mouse_points(event, x, y, flags, param):
    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        # elif len(mouse_pts) < 7:
        #     cv2.circle(image, (x, y), 5, (255, 0, 0), 10)

        if 1 <= len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts) - 1][0], mouse_pts[len(mouse_pts) - 1][1]), (70, 70, 70),
                     2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)

        if "mouse_pts" not in globals():
            mouse_pts = []

        mouse_pts.append((x, y))
        # print("Point detected")
        # print(mouse_pts)


def run(args):
    count = 0
    vs = cv2.VideoCapture(args.video_path)

    if vs is None or not vs.isOpened():
        raise Exception('Unable to open video')

    # Get video height, width and fps
    H = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps_ = int(vs.get(cv2.CAP_PROP_FPS))

    # Set scale for birds eye view
    # Bird's eye view will only show ROI
    bw_width = 400
    bw_height = 600
    scale_w = float(bw_width / W)
    scale_h = float(bw_height / H)

    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        normal_view_path = os.path.join(args.output, "normal_view.avi")
        bird_view_path = os.path.join(args.output, "bird_view.avi")
        output_movie = cv2.VideoWriter(normal_view_path, fourcc, fps_, (int(W), int(H)))
        bird_movie = cv2.VideoWriter(bird_view_path, fourcc, fps_, (bw_width, bw_height))

    json_path = os.path.join(args.output, "output.json")

    fps = FPS().start()

    global image

    persons = {}

    while vs.isOpened():

        frame_time = vs.get(cv2.CAP_PROP_POS_MSEC)

        print("Frame {} - {} ms".format(count, frame_time))

        (grabbed, frame) = vs.read()

        if not grabbed:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes = []
        head_poses = []
        ages = []
        genders = []
        ids = []

        if count == 0:
            image = frame.copy()
            while True:
                cv2.imshow("image", image)
                key = cv2.waitKey(1)

                if key & 0xFF == ord('r'):
                    image = frame.copy()
                    mouse_pts.clear()

                if key & 0xFF == ord('s'):
                    cv2.destroyWindow("image")
                    break

            src = np.float32(np.array(mouse_pts[:4]))
            dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
            prespective_transform = cv2.getPerspectiveTransform(src, dst)

            # pts = np.float32(np.array([mouse_pts[4:]]))
            # warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]
            #
            # distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
            # distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
            zone_pts = np.array(mouse_pts[:4], np.int32)

        cv2.polylines(frame, [zone_pts], True, (70, 70, 70), thickness=2)

        pilimg = Image.fromarray(rgb)
        detections = detect_image(pilimg)

        pad_x = max(rgb.shape[0] - rgb.shape[1], 0) * (img_size / max(rgb.shape))
        pad_y = max(rgb.shape[1] - rgb.shape[0], 0) * (img_size / max(rgb.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        if detections is not None:

            # persons = (detections[:, 6] == 0.)
            # detections = detections[persons]

            tracked_objects = mot_tracker.update(detections.cpu())

            for x1, y1, x2, y2, oid, cls_pred in tracked_objects:
                oid = int(oid)

                box_h = int(((y2 - y1) / unpad_h) * rgb.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * rgb.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * rgb.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * rgb.shape[1])

                boxes.append([x1, y1, box_w, box_h])
                ids.append(oid)

                color = colors[oid % len(colors)]
                color = [i * 255 for i in color]

                cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 1)
                cv2.putText(frame, str(oid), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            color, 1)

                face_x_min = int(x1 + box_w * face_w_min)
                face_x_max = int(x1 + box_w * face_w_max)
                face_y_min = y1
                face_y_max = int(y1 + box_h * face_h_max)
                face_height = abs(face_y_max - face_y_min)

                face = rgb[face_y_min:face_y_max, face_x_min:face_x_max]

                yaw, pitch, roll = estimate_head_pose(face)
                head_poses.append(["{:.2f}".format(yaw), "{:.2f}".format(pitch), "{:.2f}".format(roll)])

                draw_axis(frame, yaw, pitch, roll, tdx=(face_x_min + face_x_max) / 2,
                          tdy=(face_y_min + face_y_max) / 2, size=face_height / 2)

                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                age = estimate_age(blob)
                ages.append(age)
                gender = estimate_gender(blob)
                genders.append(gender)

                if oid not in persons.keys():
                    persons[oid] = {}
                    persons[oid]['bottom_points'] = []
                    persons[oid]['bv_points'] = []
                    persons[oid]['times'] = []
                    persons[oid]['head_poses'] = []
                    persons[oid]['genders'] = []
                    persons[oid]['ages'] = []
                    persons[oid]['gender'] = ''
                    persons[oid]['age'] = ''

                cv2.putText(frame, persons[oid]['age'], (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                cv2.putText(frame, persons[oid]['gender'], (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

                # cv2.imshow('Face', face)
                # cv2.waitKey(0)

        bv_points, bottom_points = utils.get_transformed_points(boxes, prespective_transform)

        for i, oid in enumerate(ids):
            persons[oid]['bottom_points'].append(["{:.4f}".format(bottom_points[i][0]/W), "{:.4f}".format(bottom_points[i][1]/H)])
            persons[oid]['bv_points'].append(["{:.4f}".format(bv_points[i][0]/W), "{:.4f}".format(bv_points[i][1]/H)])
            persons[oid]['times'].append("{:.2f}".format(frame_time))
            persons[oid]['head_poses'].append(head_poses[i])
            persons[oid]['ages'].append(ages[i])
            persons[oid]['genders'].append(genders[i])

            persons[oid]['age'] = max(set(persons[oid]['ages']), key=persons[oid]['ages'].count)
            persons[oid]['gender'] = max(set(persons[oid]['genders']), key=persons[oid]['genders'].count)

        bird_image = plot.bird_eye_view(bv_points, ids, bw_width, bw_height, scale_w, scale_h, colors)

        if count != 0:
            if args.save_video:
                output_movie.write(frame)
                bird_movie.write(bird_image)

            if args.show_video:
                cv2.imshow('Bird Eye View', bird_image)
                cv2.imshow('Frame', frame)

            with open(json_path, 'w') as fp:
                json.dump(persons, fp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        fps.update()
        fps.stop()

        print("[INFO] Elapsed time: {:.2f} sec".format(fps.elapsed()))
        print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

        count += 1

    if args.save_video:
        output_movie.release()
        bird_movie.release()

    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # Receives arguements specified by user
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--video', action='store', dest='video_path', default='./data/example.mp4',
                        help='Path for input video')

    parser.add_argument('-o', '--output', action='store', dest='output', default='./output/',
                        help='Path for outputs directory')

    parser.add_argument('-m', '--model', action='store', dest='model', default='./models/',
                        help='Path for models directory')

    parser.add_argument('-d', '--show_video', dest='show_video', action='store_true', default=False,
                        help='If this option is used, output video will be displayed')
    parser.add_argument('-s', '--save_video', dest='save_video', action='store_true', default=False,
                        help='If this option is used, output video will be saved')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # detection model
    detection_model, class_names = get_detection_model(args.model)
    Tensor = torch.FloatTensor

    # head pose model
    head_pose_model = get_head_pose_model(args.model)
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor)

    if torch.cuda.is_available():
        Tensor.cude()
        detection_model.cuda()
        idx_tensor.cude()
        head_pose_model.cude()

    # age model
    age_model = get_age_model(args.model)

    # gender model
    gender_model = get_gender_model(args.model)

    # tracker
    mot_tracker = Sort()

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_mouse_points)
    np.random.seed(42)

    run(args)
