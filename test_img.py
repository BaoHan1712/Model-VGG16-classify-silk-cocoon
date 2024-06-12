import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.preprocessing import image as keras_image
import os

# Định nghĩa các lớp
class_names = ['kenchuan', 'kenvang', 'kenchet', '0000']

def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    # Dong bang cac layer
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # Tao model
    input = Input(shape=(128, 128, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    # Them cac layer FC va Dropout
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation='softmax', name='predictions')(x)


    # Tạo mô hình
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

# Tải mô hình với trọng số đã huấn luyện
my_model = get_model()
my_model.load_weights("vggmodel.h5")

# Đường dẫn đến ảnh
img_path = 'image copy.png'

# Đọc ảnh
image_org = cv2.imread(img_path)

# Kiểm tra nếu ảnh được đọc thành công
if image_org is None:
    print(f"Error: Unable to load the image {img_path}.")
    exit()

# Resize ảnh
image_org = cv2.resize(image_org, dsize=(128, 128))
image = image_org.astype('float') / 255.0
image = np.expand_dims(image, axis=0)

# Predict
predict = my_model.predict(image)
print("This picture is: ", class_names[np.argmax(predict[0])], predict[0])
print(np.max(predict[0], axis=0))

if np.max(predict) >= 0.8 and np.argmax(predict[0]) != 0:
    # Show image
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1.5
    color = (0, 255, 0)
    thickness = 2
    cv2.putText(image_org, class_names[np.argmax(predict)], org, font,
                fontScale, color, thickness, cv2.LINE_AA)

# Hiển thị ảnh với kết quả dự đoán
cv2.imshow("Picture", image_org)
cv2.waitKey(0)
cv2.destroyAllWindows()
