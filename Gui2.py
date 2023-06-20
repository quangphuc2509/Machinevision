from tkinter import *
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import serial
from tkinter import font

# Define the serial port and baud rate for the HC-05 module
port = "COM1"  # Kênh port cho kênh bluetooth kết nôí máy tính 
baud_rate = 9600  # Baudrate của Protocol 

# Khởi tạo serial tên bluetooth
bluetooth = serial.Serial(port, baud_rate)

# Khởi tạo object hand bằng thư viện mediapipe
# Tạo truy suất vào module hands nằm trong solution của thư viện mediaPipe
mpHands = mp.solutions.hands
# Tạo một biến đại diện cho class Hands ( bao gồm nhận diện và theo dỏi các ngón tay)
hands = mpHands.Hands()

# drawing_utils là module trong thư viện hỗ trợ vẽ ra các đốt và các đường nối
mpDraw = mp.solutions.drawing_utils

# Tạo ra một class mới kế thừa từ lớp Top level. Khi kế thừa Toplevel, lớp con SecondWindow có thể sử dụng tất cả các phương thức 
# và thuộc tính của Toplevel. Điều này bao gồm các phương thức để quản lý và tương tác với cửa sổ như thiết lập tiêu đề, kích thước, vị trí, hiển thị, ẩn, đóng cửa sổ, và nhiều hơn nữa.
class SecondWindow(Toplevel):
    def __init__(self):
        super().__init__()
        self.video = cv2.VideoCapture(0)
        canvas_w = self.video.get(cv2.CAP_PROP_FRAME_WIDTH) * 1.5
        canvas_h = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) * 1.5
        canvas_w_win = canvas_w + 200
        canvas_h_win = canvas_h + 100
        self.geometry("{:d}x{:d}+500+100".format(int(canvas_w_win), int(canvas_h_win)))

        # Biến hiển thị chế độ điều khiển hiện tại của xe
        self.mode_var = StringVar()
        # Biến hiển thị cấp tốc độ hiện tại của xe
        self.level_var = StringVar()

        # Biến canvas dùng để chứa hình ảnh hiển thị của cam
        self.canvas = Canvas(self, width=canvas_w, height=canvas_h, bd=0)
        self.canvas.place(x=30, y=70)

        # 1 frame con chứa thông số hiển thị của chế độ chạy của xe trên tkinter
        self.chedo_frame = Frame(self, width=130, height=80, bg= "white", bd=1)
        self.chedo_frame.place(x=1000, y=100)
        # Label hiển thị chế độ chạy của xe
        self.chedo_lbl = Label(self.chedo_frame, text="Chế độ:", font=font.Font(family= "Times New Roman", weight= "bold", size=17), bg="white", bd=0, highlightthickness=0)
        self.chedo_lbl.place(x=15, y = 10)
        # Biến chứa trạng thái chế độ chạy của xe
        self.chedo_var = Label(self.chedo_frame, textvariable= self.mode_var, font=font.Font(family= "Times New Roman", weight= "bold", size=17), bg="white", bd=0, highlightthickness=0)
        self.chedo_var.place(x=15,y=40)

        # 1 frame con chứa thông số hiển thị của cấp tốc độ chạy của xe trên tkinter
        self.capdo_frame = Frame(self, width=130, height=80, bg= "white", bd=1)
        self.capdo_frame.place(x=1000, y=250)
        # Label hiển thị cấp tốc độ chạy của xe
        self.capdo_lbl = Label(self.capdo_frame, text="Cấp độ:", font=font.Font(family= "Times New Roman", weight= "bold", size=17), bg="white", bd=0, highlightthickness=0)
        self.capdo_lbl.place(x=15, y = 10)
        # Biến chứa trạng thái cấp tốc độ chạy của xe
        self.capdo_var = Label(self.capdo_frame, textvariable= self.level_var, font=font.Font(family= "Times New Roman", weight= "bold", size=17), bg="white", bd=0, highlightthickness=0)
        self.capdo_var.place(x=15,y=40)

        # Nút tắt camera
        self.stopCam_btn = Button(self, text='STOP', font=font.Font(family= "Times New Roman", weight= "bold", size=17), command=self.stopCam)
        self.stopCam_btn.place(x=1000, y =600)
        # Nút đóng của sổ điều khiển
        self.close_btn = Button(self, text='CLOSE', font=font.Font(family= "Times New Roman", weight= "bold", size=17), command=self.close)
        self.close_btn.place(x=1000, y= 700)
        # Nút bật camera
        self.openCam_btn = Button(self, text='OPEN', font=font.Font(family= "Times New Roman", weight= "bold", size=17), command= self.open)
        self.openCam_btn.place(x=1000, y= 500)
        
        # Biến chứa ảnh định dạng dành cho tkinter
        self.photo = None
    
    # Hàm này xác định ngón tay nào đươc trỏ lên
    # Từ ngón trỏ tới ngón út sẽ xác định sự giơ lên bằng cách so sánh tọa độ y của đầu ngón và cuối đốt của ngón 
    # Riêng ngón cái sẽ xác định sự trỏ ra bằng cách xét theo phương x. 
    # chú ý (Khi xét theo phương x thì sự trỏ ra của x sẽ khác nhau giưã tay trái và phải)
    # Tóm tắt: Xác định ngón. Ngón (2->5) theo y và ngón (1) theo x 
    # Input của hàm sẽ là mảng chứa giá trị tọa độ của các ngón trên bàn tay -> hand_landmarks, biến chứa giá trị tay trái hay phải Type_hand
    def detect_finger(self, hand_landmarks, Type_hand):
        # Tạo mảng A để chứa các trạng thái của ngón tay
        # Mảng này sẽ chứ True hoặc False (nếu trạng thái ngón tay lên là True và ngược lại)
        A = {}
        # 1. Xét trạng thái tay trái hay phải
        # Nếu là tay trái
        for hand in hand_landmarks:
            if(Type_hand == 'Left'):
                # Biến chỉ tay trái hay phải (Trái là True)
                A[0] = True
                # Các biến hand.landmark[i] ở đây tương ứng với các đốt của ngón tay tương ứng được tra theo list của thư viện 
                # Ngon ut
                tip_5 = hand.landmark[20]
                pip_5 = hand.landmark[19]
                A[5] = tip_5.y < pip_5.y    # So sánh đầu ngón và cuối ngón, nếu nhỏ hơn trả về True 

                # Ngon ap ut
                tip_4 = hand.landmark[16]
                pip_4 = hand.landmark[14]
                A[4] = tip_4.y < pip_4.y

                # Ngon giua
                tip_3 = hand.landmark[12]
                pip_3 = hand.landmark[10]
                A[3] = tip_3.y < pip_3.y

                # Ngon tro
                tip_2 = hand.landmark[8]
                pip_2 = hand.landmark[6]
                A[2] = tip_2.y < pip_2.y

                # Ngon cai
                tip_1 = hand.landmark[4]
                pip_1 = hand.landmark[3]
                A[1] = tip_1.x > pip_1.x

            elif(Type_hand == 'Right'):
                A[0] = False

                # Ngon ut
                tip_5 = hand.landmark[20]
                pip_5 = hand.landmark[19]
                A[10] = tip_5.y < pip_5.y

                # Ngon ap ut
                tip_4 = hand.landmark[16]
                pip_4 = hand.landmark[14]
                A[9] = tip_4.y < pip_4.y

                # Ngon giua
                tip_3 = hand.landmark[12]
                pip_3 = hand.landmark[10]
                A[8] = tip_3.y < pip_3.y

                # Ngon tro
                tip_2 = hand.landmark[8]
                pip_2 = hand.landmark[6]
                A[7] = tip_2.y < pip_2.y

                # Ngon cai
                tip_1 = hand.landmark[4]
                pip_1 = hand.landmark[3]
                A[6] = tip_1.x < pip_1.x

        # Trả về các giá trị theo biến A
        # List: 
        # A[0] Tay trái hay phải
        # A[1 -> 5] Các ngón bên tay trái
        # A[6 -> 10] Các ngón bên tay phải 
        return A
    
    # Hàm này sẽ chuyển các tín hiệu từ ngón tay thành các lệnh để gửi bluetooth về cho xe
    def Ma_hoa_thanh_lenh(self, hand_landmarks):
        hand_type = self.results.multi_handedness[0].classification[0].label  # Nhận dạng tay trái hay phải
        Dinh_dang = self.detect_finger(hand_landmarks, hand_type)
        # Biến dùng để trả về giá trị chế độ hoạt động của xe
        encode_var_mode = 'Dừng'
        # Biến dùng để trả về giá trị cấp tốc độ của xe
        encode_var_level = '1'
        # Biến kiểm tra tay điểu khiển tốc độ của xe có đang được nhận dạng không =
        check = False
        # Nếu là tay trái 
        if(Dinh_dang[0]  == True):
                # Tien len
            if(Dinh_dang[2] == True and Dinh_dang[1] == False and Dinh_dang[5] == False):
                Encode_data = 'F0'
                encode_var_mode = 'Tiến'
                # Tien len Trai
            elif(Dinh_dang[2] == True and Dinh_dang[1] == True and Dinh_dang[5] == False):
                Encode_data = 'FL'
                encode_var_mode = 'Tiến Trái'
                # Tien len Phai
            elif(Dinh_dang[2] == True and Dinh_dang[1] == False and Dinh_dang[5] == True):
                Encode_data = 'FR'
                encode_var_mode = 'Tiến Phải'
                # Lui 
            elif(Dinh_dang[2] == False and Dinh_dang[1] == False and Dinh_dang[5] == False):
                Encode_data = 'B0'
                encode_var_mode = 'Lùi'
                # Lui Phai
            elif(Dinh_dang[2] == False and Dinh_dang[1] == False and Dinh_dang[5] == True):
                Encode_data = 'BR'
                encode_var_mode = 'Lùi Phải'
                # Lui Trai
            elif(Dinh_dang[2] == False and Dinh_dang[1] == True and Dinh_dang[5] == False):
                Encode_data = 'BL'
                encode_var_mode = 'Lùi Trái'
                # Luon luon dung lai khi khong co ki tu nao
            else:
                Encode_data = 'ST'
                encode_var_mode = 'Dừng'

        # Nếu là tay phải 
        if(Dinh_dang[0]  == False):
            check = True
        ##########################
            if(Dinh_dang[7] == True and Dinh_dang[8] == False and Dinh_dang[9] == False):
                Encode_data = 'L1'
                encode_var_level = '1'
            elif(Dinh_dang[7] == True and Dinh_dang[8] == True and Dinh_dang[9] == False):
                Encode_data = 'L2'
                encode_var_level = '2'
            elif(Dinh_dang[7] == True and Dinh_dang[8] == True and Dinh_dang[9] == True):
                Encode_data = 'L3'
                encode_var_level = '3'
            else:
                Encode_data = 'ST'
                check = False
        #########################
        return Encode_data, encode_var_mode, encode_var_level, check

    # Hàm thể hiện thông số lên giao diện và gửi giá trị nhận được đến hc05
    def interface_output(self, frame, Ma_lenh):
        # Tien len
        if(Ma_lenh == 'F0'):
            cv2.putText(frame, "^", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            bluetooth.write(Ma_lenh.encode())
            # Tien len Trai
        elif(Ma_lenh == 'FL'):
            cv2.putText(frame, "<-^", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            bluetooth.write(Ma_lenh.encode())
            # Tien len Phai
        elif(Ma_lenh == 'FR'):
            cv2.putText(frame, "^->", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            bluetooth.write(Ma_lenh.encode())
            # Lui 
        elif(Ma_lenh == 'B0'):
            cv2.putText(frame, "v", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            bluetooth.write(Ma_lenh.encode())
            # Lui Phai
        elif(Ma_lenh == 'BR'):
            cv2.putText(frame, "v->", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            bluetooth.write(Ma_lenh.encode())
            # Lui Trai
        elif(Ma_lenh == 'BL'):
            cv2.putText(frame, "<-v", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            bluetooth.write(Ma_lenh.encode())
        #--------------------------------------------------
        elif(Ma_lenh == 'L1'):    # Level PWM 01
            cv2.putText(frame, "01", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            bluetooth.write(Ma_lenh.encode())
        elif(Ma_lenh == 'L2'):    # Level PWM 02
            cv2.putText(frame, "02", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            bluetooth.write(Ma_lenh.encode())
        elif(Ma_lenh == 'L3'):         # Level PWM 03
            cv2.putText(frame, "03", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            bluetooth.write(Ma_lenh.encode())
        #--------------------------------------------------
        # Luôn luôn dừng lại khi không được phát hiện được tay
        else:
            cv2.putText(frame, "STOP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            bluetooth.write(Ma_lenh.encode())

    def update_frame(self):
        ret, frame = self.video.read()
        
        frame = cv2.resize(frame, dsize=None, fx=1.5, fy=1.5)
        # Bởi vì CV2 nó trả về BGR mà ảnh process bởi thư viện yêu cầu là RGB nên phải convert lại
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mode_var = 'Dừng'
        level_var = ''

        # Xử lí ảnh đối tượng bàn tay
        # Tất cả các phương cách xử lí được lưu vào biến results (tọa độ các đốt ngón tay, tay trái hay phải, đường biên tọa độ...)
        self.results = hands.process(frame)

        # Nó sẽ lưu trữ tọa độ x,y,z của các đốt ngón tay theo format của list thư viện
        hand_landmarks = self.results.multi_hand_landmarks

        # Kiểm tra xem đã xác định được bàn tay hay chưa, nếu có thì sẽ processing 
        if hand_landmarks:
            # Lấy hết tất cả tọa độ của đốt ngón tay 
            for hand in hand_landmarks:
                # Quét từ 0 đến 20
                # Vẽ ra các kết nối nhờ vào drawing_utils
                # Custom lại các màu đốt và đường kết nối của bàn tay 
                mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)

            Ma_lenh, mode_var, level_var, check = self.Ma_hoa_thanh_lenh(hand_landmarks)
            print(Ma_lenh)

            self.interface_output(frame, Ma_lenh)
            
            # Cập nhật các biến self.mode_var, self.level_var
            self.mode_var.set('{}'.format(mode_var))
            if (check == True):
                # Chỉ cập nhật khi check = True (có tay điều khiển cấp tốc độ)
                self.level_var.set('{}'.format(level_var))
        else:
            # Nếu không phát hiện bàn tay nào sẽ gửi tín hiệu dừng xe
            self.mode_var.set('{}'.format(mode_var))
            cv2.putText(frame, "STOP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            bluetooth.write("ST".encode())
            print("stop")

        # Tạo ra một ảnh PIL 
        image = Image.fromarray(frame)
        # Chuyển ảnh sang định dạng ảnh của Tkinter
        self.photo = ImageTk.PhotoImage(image=image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

        # Cập nhật lại hàm update_frame mỗi 10ms
        self.after(10, self.update_frame)

    # Tạo hàm tắt webcam
    def stopCam(self):
        self.canvas.delete("all")
        self.video.release()
        
    # Tạo hàm đóng cửa sổ
    def close(self):
        self.video.release()
        self.destroy()

    # Tạo Hàm mở webcam
    def open(self):
        self.video = cv2.VideoCapture(0)
        self.after(5, self.update_frame)