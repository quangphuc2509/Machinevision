from tkinter import * # Thư viện Tkinter để thiết kế giao diện 
from tkinter import messagebox
import cv2 # Thư viện openCV để xử lý ảnh
from PIL import Image, ImageTk, ImageEnhance # Thư viện Pillow để xử lý ảnh 
import tensorflow as tf
import numpy as np
import time # Thư viện thời gian
import math  # thư viện toán học
from keras.models import load_model # thư viện dùng để lấy model đã train
from cvzone.HandTrackingModule import HandDetector  # thư viện nhận dạng tay 

from Gui2 import * # thêm vào chương trình Gui2 (là cửa sổ thứ 2)

# Danh sách các lớp (chữ cái) được nhận dạng
class_name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space']

# Chạy mô hình đã được train 
save_model = load_model("detectlanguage.hdf5")

# Khai báo biến webcam và video    
video = cv2.VideoCapture(0)

# Chiều dài chiều rộng cho cửa sổ giao diện 1
canvas_w_win = video.get(cv2.CAP_PROP_FRAME_WIDTH) * 1.5 + 200
canvas_h_win = video.get(cv2.CAP_PROP_FRAME_HEIGHT) * 1.5 + 100

# Khởi tạo HandDetector để nhận dạng tay, với tham số maxHands để chỉ ra số lượng tay tối đa nhận dạng 
detector = HandDetector(maxHands=1) 

# Các tham số cần thiết cho việc xử lý ảnh
offset = 20
imgSize = 300

# Tạo cửa sổ Tkinter
window = Tk()
window.title("tkinter")
window.geometry("{:d}x{:d}+500+100".format(int(canvas_w_win), int(canvas_h_win))) # Cửa sổ có kích thước là , canvas_h_win ở vị trí (500,100)

# Khởi tạo các biến
text = "" # Biến để lưu trữ văn bản sau khi đọc được từ webcam
start = time.time() # Biến dùng để lưu giá trị thời gian bắt đầu thực hiện quá trình
letter_old = "" # Biến dùng để lưu lại ký tự hiện tại
letter = ""     # Biến dùng để lưu lại ký tự trước
check_even = 0  # Biến trạng thái hoạt động để xác định chữ cái
interval = 0.5  # Thời gian dùng để xác định chữ cái 
pass_text = StringVar() # Biến string có giá trị thay đổi dùng để thể hiện trên Entry

# Hàm reset 
def resetfunc():
    global video, button, state, imgage_obj, canvas_img, pass_text, text # Khai báo toàn cục cho các biến cần thiết 
    video.release()
    # Ngắt trạng thái hoạt động của nút điều khiển
    button.config(state='disable')
    # Cho canvas hiển thị hình ảnh ban đầu 
    canvas_img.create_image((0,0),image = imgage_obj, anchor = NW)
    # Xóa văn bản đã lưu và xóa hiển thị văn bản đó trên Entry
    text =""
    pass_text.set("")
    # Khởi động lại biến camera
    video = cv2.VideoCapture(0)

# Hàm thực hiện việc nhận biết và phan tích
def update_frame():
    # Khai báo toàn cục cho các biến cần thiết 
    global canvas_img, photo1, video, button, imgage_obj, state, img_photo, start, check_even, interval, letter, letter_old, text, pass_text

    # Đọc hình ảnh từ camera, ret: biến kiểm tra camera có hoạt động không, frame1: biến chứa khung hình của camera
    ret, frame1 = video.read()
    # Tạo ra một khung hình mới được copy từ khung hình gốc dùng để thể hiện lên giao diện mà không bị ảnh hưởng bởi việc xử lý hình ảnh nhận dạng tay
    frame2 = frame1.copy()
    # Đưa biến trạng thái về chế độ bật cam nhận diện
    state = 0
    # Nhận dạng tay trong ảnh
    hands,_ = detector.findHands(frame1)

    # Nếu có bàn tay
    if hands:
        # hands chứa dữ liệu các bàn tay đã nhận diện được, hands[0] chứ thông tin về bàn tay đầu tiên nhận diện được
        hand = hands[0]
        # x,y,w,h lần lượt là tọa độ, chiều dài và chiều cao của bàn tay được phát hiện 
        x, y, w, h = hand['bbox']

        # imgWhite: tạo ra một mảng numpy có kích thước (imgSize, imgSize, 3), có tất cả giá trị trong mảng bằng 1 *255, 
        #           các giá trị có kiểu là uint8, hay còn gọi là tạo ra một hình ảnh màu trắng có kích thước (imgSize,imgSize)
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        # Cắt ảnh để lấy phần tay
        imgCrop = frame1[y - offset:y + h + offset, x - offset:x + w + offset]

        # Tạo biến chứa tỷ lệ của ảnh bàn tay 
        aspectRatio = h / w
        
        if aspectRatio > 1: 
            # Ảnh có hình dạng chữ nhật dọc
            k = imgSize / h  # Tính hệ số tỷ lệ kích thước
            wCal = math.ceil(k * w)  # Tính kích thước chiều rộng mới
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # Thay đổi kích thước ảnh
            wGap = math.floor((imgSize - wCal) / 2)  # Tính khoảng cách thêm vào 2 bên của ảnh
            imgWhite[:, wGap:wCal + wGap] = imgResize  # Sao chép ảnh thay đổi vào ảnh trắng

        else:
            # Ảnh có hình dạng chữ nhật ngang
            k = imgSize / w  # Tính hệ số tỷ lệ kích thước
            hCal = math.ceil(k * h)  # Tính kích thước chiều cao mới
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # Thay đổi kích thước ảnh
            hGap = math.floor((imgSize - hCal) / 2)  # Tính khoảng cách thêm vào 2 bên của ảnh
            imgWhite[hGap:hCal + hGap, :] = imgResize  # Sao chép ảnh thay đổi vào ảnh trắng

        img = cv2.resize(imgWhite, dsize=(128, 128))
        # chuẩn hóa giá trị của ảnh nằm trong phạm vi 0 đến 1, giúp cho quá trình xác định được ổn định hơn 
        # và đảm bảo sự tương đồng giữa trong quá trình so sánh
        img = img.astype('float')*1./255
        # Chuyển đổi sang tensor bằng cách mở rộng chiều 
        img = np.expand_dims(img, axis=0)

        # Dự đoán
        predict = save_model.predict(img)
        
        #---------------------------- Thêm ký tự vào văn bản -------------------------------------------------------
        # Lấy thời gian hiện tại để so sánh 
        end = time.time()
        
        # Kiểm tra thời gian đã trôi qua từ lần dự đoán trước đó và so sánh với khoảng thời gian cho trước (interval)
        # Nếu thời gian lớn hơn 0.5s và biến kiểm tra = 0 thì sẽ lưu ký tự hiện tại vào text
        if (end - start >= interval) and (check_even == 0):
            # làm mới thời điểm bắt đầu
            start = time.time()
            # Lấy ký tự có xác suất dự đoán cao nhất
            letter = class_name[np.argmax(predict[0])]
            check_even = 1

        # Kiểm tra thời gian đã trôi qua từ lần dự đoán trước đó và so sánh với khoảng thời gian cho trước (interval)
        # Nếu thời gian lớn hơn hoặc bằng 0.5s và biến kiểm tra là 1
        if (end - start >= interval) and (check_even == 1):
            check_even = 0
            # làm mới thời điểm bắt đầu
            start = time.time()
            # ký tự hiện tại 
            letter_old = class_name[np.argmax(predict[0])]
    
            # Kiểm tra ký tự dự đoán có giống ký tự trước đó không
            if letter_old == letter:
                # Nếu ký tự không phải là "del" hoặc "space", thêm ký tự vào văn bản
                if letter != "del" and letter != "space":
                    text += letter
                    
                # Nếu ký tự là "del", xóa ký tự cuối cùng trong văn bản
                elif letter == "del":
                    text = text[0:-1]
                # Nếu ký tự là "space", thêm dấu gạch dưới vào văn bản
                else:
                    text += "_"
        
        # Khoanh vùng bàn tay
        cv2.rectangle(frame2, (x- 20, y - 20),(x + w + 20, y + h + 20),(255, 0, 255), 2) 
        # Hiển thị chữ cái lên chỗ khoanh vùng 
        cv2.putText(frame2, class_name[np.argmax(predict[0])], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 255, 0), 2) 

    # Bởi vì canvas để hiển thị camera ở GUI1 chỉ có kích thước (427,320) nên phải resize lại kích thước hiển thị webcam 
    frame2 = cv2.resize(frame2, (427,320))
    # Chuyển sang hệ màu RGB vì Tkinter xài hệ màu RGB trong khi opencv sử dụng hệ màu BGR
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    # Tạo ra một ảnh PIL 
    image1 = Image.fromarray(frame2)
    # Chuyển ảnh sang định dạng ảnh của Tkinter
    photo1 = ImageTk.PhotoImage(image=image1)

    # thay đổi giá trị biến pass_text theo text
    pass_text.set('{}'.format(text))
    
    #Kiểm tra độ dài của mật khẩu nhập vào
    if (len(text)==3):
        # Tắt camera
        video.release()
        # Kiểm tra mật khẩu
        if (text == "AAA"):
            # Kích hoạt trạng thái hoạt động của nút điều khiển 
            button.config(state='active')
            state = 1
        else:
            messagebox.showerror("Lỗi", "Mật khẩu không chính xác!") # Thông báo lỗi 
        
    # Trạng thái 0: canvas hiển thị hình ảnh từ cam, trạng thái 1: đóng cam, hiển thị ảnh đã đăng nhập thành công
    if (state == 0):
        canvas_img.delete("all")
        canvas_img.create_image(0, 0, image=photo1, anchor=NW)   
    elif (state == 1):
        img_success = Image.open('success.png').resize([427,320])
        img_photo = ImageTk.PhotoImage(image= img_success)
        canvas_img.create_image((0,0),image = img_photo, anchor = NW)

    # thực hiện hàm update_frame sau mỗi 10ms    
    window.after(10, update_frame)

# Đường dẫn ảnh background
background_url = r'backgroudmain.png'
# Load ảnh
background_image = Image.open(background_url).resize((1160,820))
# Chuyển đổi hình ảnh thành định dạng hình ảnh tkinter
background_photo = ImageTk.PhotoImage(background_image)

# Tạo một nhãn với ảnh có độ trong suốt đã được giảm
background_label = Label(window, image=background_photo)
background_label.place(x=0,y=0)

# biến canvas để hiển thị camera
canvas_img = Canvas(window,  width=427, height=320)
canvas_img.place(x = 30, y =480)

# Hình ảnh canvas
image_canvas = Image.open('image.png').resize([427,320])
imgage_obj = ImageTk.PhotoImage(image_canvas)
canvas_img.create_image((0,0),image = imgage_obj, anchor = NW)

# Tạo giao diện nút nhấn 
button1 = Button(window, text="Nhận dạng",font= font.Font(family= "Times New Roman",  size=17), width=12,height=2, command=update_frame)
button1.place(x=620, y=695)
button2 = Button(window, text="RESET",font= font.Font(family= "Times New Roman",  size=17), width=12,height=2, command=resetfunc)
button2.place(x=920, y=695)
button = Button(window, text="Điều khiển",font= font.Font(family= "Times New Roman",  size=17), width=12,height=2, command=SecondWindow, 
                state='disable')
button.place(x=770, y=695)

pass_entry = Entry(window, show="*", bg="white", font= font.Font(family= "Times New Roman",  size=18), width=20, textvariable=pass_text)
pass_entry.place(x=620,y= 770)

photo1 = None

window.mainloop()
