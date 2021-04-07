from ctypes import *                                  # ctypes는 dll에서 제공하는 함수 호출을 한다.
import math                                           # 연산에 관련한 math 모듈을 import한다.
import random                                         #random 모듈을 import한다.
import os

#다음의 class들은 모두 Structure를 상속받은 class들이다.
# 이 class들은 ctype을 이용해 변환 시켜준다.

class BOX(Structure):                                 # 각 grid cell은 2개의 Bounding box를 예측하고 예측값을 갖는다.
    _fields_ = [("x", c_float),                       # x, y는 중심점의 좌표이다.
                ("y", c_float),
                ("w", c_float),                       # w는 너비를 나타낸다.
                ("h", c_float)]                       # h는 높이를 나타낸다.
 

class DETECTION(Structure):                           
    _fields_ = [("bbox", BOX),                        # Bounding BOX 이미지를 찾을때 사용하는 경계상자
                ("classes", c_int),                   # 클래스의 수를 나타낸다.
                ("prob", POINTER(c_float)),           # prob array 확률을 나타낸다.
                ("mask", POINTER(c_float)),           # 사용될 tag에 해당하는 anchor를 나타낸다.
                ("objectness", c_float),              # 각 box는 객체의 위치 (x, y), 객체의 크기 (w, h), box confidence score 5개를 갖게 된다.
                                                      # 이때 Box confidence score가 Objectness를 나타낸다.
                ("sort_class", c_int),                # class를 정렬한다.
                ("uc", POINTER(c_float)),             # RGB-D 돌출성 검출에 대한 uncertainity을 나타낸다.
                ("points", c_int),                    # 객체 포인트들을 나타낸다.
                ("embeddings", POINTER(c_float)),     # 카테고리간의 개념과 관련도를 시각화 해주는 용도
                ("embedding_size", c_int),            # embedding의 크기를 나타낸다.
                ("sim", c_float),                     # 객체 유사도를 나타낸다.
                ("track_id", c_int)]                  # 객체 추적을 의미한다. 

class DETNUMPAIR(Structure):                          # 포인터를 감지한다.
    _fields_ = [("num", c_int),                      
                ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),                         # w는 너비를 나타낸다.  
                ("h", c_int),                         # h는 높이를 나타낸다.
                ("c", c_int),                         # c는 중심점(centroid)을 나타낸다.
                ("data", POINTER(c_float))]           # 데이터의 포인터를 나타낸다.


class METADATA(Structure):                            # 데이터에 관한 구조화된 데이터로, 다른 데이터를 설명해 주는 데이터이다.
    _fields_ = [("classes", c_int),        
                ("names", POINTER(c_char_p))]         # 객체의 수와 이름을 나타낸다.


def network_width(net):                                                            # 네트워크의 너비 width은 레이어의 최대 노드 수로 정의된다.
    return lib.network_width(net)


def network_height(net):                                                           # 네트워크의 깊이 height는 계층의 수로 정의된다. 
    return lib.network_height(net)


def bbox2points(bbox):                                                             # 욜로 형식의 bbox에서부터 코너 포인트 cv2 직사각형까지

    x, y, w, h = bbox                                                              # bbox의 x,y,w,h 값을 나타낸다.
    xmin = int(round(x - (w / 2)))                                                 # x의 최소 코너 지점을 계산한다.
    xmax = int(round(x + (w / 2)))                                                 # x의 최대 코너 지점을 계산한다.
    ymin = int(round(y - (h / 2)))                                                 # y의 최소 코너 지점을 계산한다. 
    ymax = int(round(y + (h / 2)))                                                 # y의 최대 코너 지점을 계산한다.
    return xmin, ymin, xmax, ymax                                                  # 계산된 값들을 return 시킨다.


def class_colors(names):                                                           # 각 클래스 이름에 대해 임의의 RGB 색상 하나를 사용하여 dict을 만든다.

    return {name: (                                                                # 각각의 개체에 RGB 색상을 정해준다.
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}


def load_network(config_file, data_file, weights, batch_size=1):                   # batch_size=1 은 한번에 한장의 사진을 처리한다.

    network = load_net_custom(
        config_file.encode("ascii"),
        weights.encode("ascii"), 0, batch_size)                                    # 입력요소를 ascii 형식의 Byte type으로 가져온다.
    metadata = load_meta(data_file.encode("ascii"))                                # METADATA를 ascii 형식의 Byte type으로 가져온다.
    class_names = [metadata.names[i].decode("ascii") for i inrange(metadata.classes)]
    colors = class_colors(class_names)                                             # claas_colors 함수를 통해 색상을 지정한다.
    return network, class_names, colors                                            # network, class_names, colors를 반환한다.


def print_detections(detections, coordinates=False):
    print("\nObjects:")
    for label, confidence, bbox in detections:                                     # detections 안에 label, confidence, bbox를 for loop를 통해 반복시켜준다.
        x, y, w, h = bbox                                                          # bbox 반환값을 지정해준다.
        if coordinates:                                                            # coordinates=True 이면
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))  # 객체 이름과 신뢰도 bounding box를 나타내준다.
        else:
            print("{}: {}%".format(label, confidence))                             # coodinates=False 이면 객체이름과 신뢰도만 표시해준다.


def draw_boxes(detections, image, colors):                                         # 감지된 이미지에 컬러 박스를 친다.
    import cv2                                                                     # open cv2를 import 한다.
    for label, confidence, bbox in detections:                                     # label, confidence, bbox를 bbox 포인트로 사용한다.
        left, top, right, bottom = bbox2points(bbox)                               # 왼쪽 위, 오른쪽 아래를 포인트로 잡는다.
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)       # 네모난 박스를 친다.     
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),         # 박스에 text를 삽입한다.
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image                                                                   # 이미지를 리턴한다.
   

def decode_detection(detections):
    decoded = []                                                                   # decoded array를 선언 및 초기화한다.                                                          
    for label, confidence, bbox in detections:                                     # detections 안에 label, confidence, bbox를 for loop를 통해 반복시켜준다.
        confidence = str(round(confidence * 100, 2))                               # 신뢰도는 round 함수를 통해 구해준다.
        decoded.append((str(label), confidence, bbox))                             # decoded 딕셔너리에 추가시켜준다.
    return decoded                                                                 # decoded를 반환해준다.


def remove_negatives(detections, class_names, num):                                # 감지 내에서 0% 신뢰도인 모든 클래스를 제거한다.

    predictions = []                                                               # predictions array를 선언 및 초기화한다.
    for j in range(num):
        for idx, name in enumerate(class_names):
            if detections[j].prob[idx] > 0:
                bbox = detections[j].bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                predictions.append((name, detections[j].prob[idx], (bbox)))        # predictions에 name, prob, detection을 추가한다.
    return predictions                                                             # predictions을 반환한다.


def detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45): # 신뢰도가 가장 높은 목록과 해당 bbox를 반환한다.

    pnum = pointer(c_int(0))
    predict_image(network, image)
    detections = get_network_boxes(network, image.w, image.h,
                                   thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(detections, num, len(class_names), nms)
    predictions = remove_negatives(detections, class_names, num)                   # remove_negatives 함수를 이용하여 prediction의 내에서 0% 신뢰도인 모든 클래스를 제거한다.
    predictions = decode_detection(predictions)                                    # detection(predictions)을 str형으로 바꿔준다.
    free_detections(detections, num)                                               
    return sorted(predictions, key=lambda x: x[1])                                 


#  lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)      컴파일 하면 libdarknet.so 파일이 생성되고, CDLL을 이용하여 모듈을 로딩한다.
#  lib = CDLL("libdarknet.so", RTLD_GLOBAL)
hasGPU = True                                                                      
if os.name == "nt"                                                                 # 만약os 모듈이 windows에서 실행된다면
    cwd = os.path.dirname(__file__)                                                # 파일의 폴더 경로를 리턴한다.
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']                            # 폴더 경로와 환경변수를 더해준다.
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")                              # path들을 묶어 하나의 경로로 만들어준다.
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")                      # path들을 묶어 하나의 경로로 만들어준다.
    envKeys = list()                                                               # envKeys는 리스트화 시켜준다.
    for k, v in os.environ.items():                                                # os.environ모듈에 Key와 Value를 for loop를 통해 입력해준다.
        envKeys.append(k)                                                          # Key값을 for loop동안 추가시켜준다.
    try:                                                                           # try 블록 수행중 오류 발생시 except블록에서 대신 수행한다.
        try:                                                                       # try 블록 수행중 오류 발생시 except블록에서 대신 수행한다.
            tmp = os.environ["FORCE_CPU"].lower()                                  # 환경 변수에 저장된 값을 소문자로 읽어온다.
            if tmp in ["1", "true", "yes", "on"]:                                  # 만약 1,true,yes,on이 tmp에 있다면
                raise ValueError("ForceCPU")                                       # ValueError를 나타낸다. 
            else:                                                                  # 그렇지 않다면 
                print("Flag value {} not forcing CPU mode".format(tmp))            # tmp 값을 프린트 시켜준다.
        except KeyError:                                                           # try에서 KeyError발생시 수행한다.
          # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:                                  # envKeys에 문자가 있다면
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:                    # os.environ[문자] 정수값이 0보다 작다면
                    raise ValueError("ForceCPU")                                   # ValueError가 발생한다.
            try:                                                                   # try 블록 수행중 오류 발생시 except블록에서 대신 수행한다. 
                global DARKNET_FORCE_CPU                                           # 전역변수를 설정해준다.
                if DARKNET_FORCE_CPU:                                              # 만약 전역변수가 온다면
                    raise ValueError("ForceCPU")                                   # ValueError를 나타낸다.
            except NameError as cpu_error:                                         # NameError의 오류 메시지를 알려준다. 
                print(cpu_error)                                                   # 오류 메시지를 출력한다. 
        if not os.path.exists(winGPUdll):                                          # winGPUdll가 경로에 존재하지 않는다면
            raise ValueError("NoDLL")                                              # ValueError를 나타낸다.
        lib = CDLL(winGPUdll, RTLD_GLOBAL)                                         # CDLL을 통해 DLL을 로드시키고 lib에 지정해준다.
    except (KeyError, ValueError):                                                 # KeyError,ValueError오류 발생시 수행한다.
        hasGPU = False                                           
        if os.path.exists(winNoGPUdll):                                            # 경로에 winNoGPUdll이 존재한다면
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)                                   # CDLL을 통해 DLL을 로드시키고 lib에 지정해준다
            print("Notice: CPU-only mode")                                         # mode를 출력해준다.
        else:                                                                      # 경로에 winNoGPUdll이 존재하지 않는다면   
            # Try the other way, in case no_gpu was compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)                                     # CDLL을 통해 DLL을 로드시키고 lib에 지정해준다
            print("Environment variables indicated a CPU run, but we didn't find {}. Trying a GPU run anyway.".format(winNoGPUdll))            # 오류를 출력해준다.
else:#os 모듈이 windows에서 실행되지 않는다면
    lib = CDLL(os.path.join(os.environ.get('DARKNET_PATH', './'),"libdarknet.so"), RTLD_GLOBAL) # CDLL을 통해 DLL(환경에서 얻어낸 경로를 하나의 경로로 만든)을 로드시키고 lib에 지정해준다.
    
lib.network_width.argtypes = [c_void_p]                                            
lib.network_width.restype = c_int                                                  # 밑에 부분은 ctype 함수를 이용해 C언어 데이터를 파이썬에 맞는 type으로 바꿔준다.
lib.network_height.argtypes = [c_void_p]                                           
lib.network_height.restype = c_int                                                
 
copy_image_from_bytes = lib.copy_image_from_bytes                                  
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]                                  

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:                                                                         
    set_gpu = lib.cuda_set_device 
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

free_network_ptr = lib.free_network_ptr
free_network_ptr.argtypes = [c_void_p]
free_network_ptr.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                   c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DETNUMPAIR)
