# Import necessary libraries/modules

"""
概念介绍
1.OpenCV(cv2):
    用于图像处理和计算机视觉任务的库
    包含面部检测模型（如Haar级联分类器）
2.Streamlit:
    用于构建和部署数据驱动应用的开源框架
    提供了渐变的用户界面创建方法
3.Keras和TensorFlow:
    用于构建和训练深度学习模型的库
    `model_from_json`和`load_weights`用于加载预训练模型
4.WebRTC和streamlit_webrtc:
    实时通信协议，用于视频流和音频流传输
    `streamlit_webrtc`扩展库继承了WebRTC功能，使其能与Streamlit应用结合
5.面部检测和情绪识别:
    使用Haar级联分类器检测面部
    使用预训练的卷积神经网络（CNN）模型进行情绪识别
6.级联分类器:
    级联分类器是一种级联结构的检测器，通常用于对象检测任务，例如面部检测。Haar级联分类器是一种常见的级联分类器。
    级联分类器的基本思想是将一系列简单的分类器（弱分类器）按顺序排列，每个分类器用于逐步细化检测结果。
        级联:指的是一系列的分类器按顺序排列，前一个分类器的输出将作为后一个分类器的输入。
        通过这种方式可以快速排查大量的负例（即非目标对象），从而减少计算负担。
        只有通过所有级联分类器的候选区域才会被认为是目标对象。
7.图像尺寸减少的比例（scaleFactor）:
    指在多尺度检测过程中，每次缩小图像的比例。Haar级联分类器使用滑动窗口来检测图像中的目标对象。
    为了检测不同尺寸的目标（例如不同大小的面部），需要对图像进行多尺度处理。
    - scaleFactor:控制每次图像尺寸减少的比例。例如，如果`scaleFactor`设置为1.3，那么图像尺寸将每次减少30%
                  具体来说，如果原始图像的尺寸为1000*1000像素，那么下一次检测时的图像尺寸将变为1000/1.3≈769像素
                  通过这种方法，可以检测不同大小的面部。
8.候选矩形保留之前检测所需的临近框数（minNeighbors）
    指在保留一个候选区域之前，该地区需要被多少个邻近的检测框覆盖，这有助于误检。
    minNeighbors:指定每个候选矩形（可能是面部区域）在被认为是最终检测结果之前所需的最低邻近框数。
                 一个候选区区域需要至少有 `minNeighbors` 个检测框重叠才能被保留为最终的检测结果。
                 如：如果 `minNeighbors` 设置为5，只有那些被至少5个检测框覆盖的候选区域才会被保留为最终的检测结果，
                 有助于过滤掉噪声和误检。

为什么要这样做？
    面部检测和绘制矩形框:
    可视化检测到的面部，使用户直观地看到面部检测结果。

    ROI提取和调整大小:
    确保提取的面部区域符合情绪识别模型的输入要求。

    标准化处理:
    归一化图像像素值，提高神经网络模型的预测性能。

    情绪预测:
    使用预训练的模型进行情绪识别，为每个检测到的面部提供实时的情绪反馈。

    显示预测结果:
    将情绪标签显示在检测到的面部上，提供用户友好的界面和即时反馈。

为什么需要进行归一化处理:
    1.标准化输入数据:神经网络通常对输入数据的范围和分布非常敏感。将数据标准化有助于似的不同特征具有相同的尺寸，
                   从而加速训练过程，提高模型的手链速度，并且肯呢个提高模型的预测性能。
    2.数值稳定性:大多数现代深度学习框架和模型都是在数值稳定性和效率的前提下设计的。
                标准化输入数据可以避免大数值引起的数值溢出或精度问题。将数据归一化到0-1范围内可以帮助保持数值计算的稳定性。
    3.适应激活函数的输入范围:很多激活函数(如Sigmoid、Tanh、ReLU等)在输入值接近特定范围时表现最佳。如Sinmoid函数的输出
                         在输入接近0时变化最为剧烈，而输入过大或过小时会变换的非常平缓。因此，归一化可以确保输入值在
                         激活函数的敏感范围内，使得模型更容易学习到有效特征。
    4.一致的数据处理:将输入数据标准化为一个一致的范围有助于模型在不同数据集或不同输入情况下表现一致。
                  这种一致性对模型的训练和预测非常重要，可以减少模型在不同数据分布上的性能波动。
"""

# numpy:一个用于数组处理的库
import numpy as np
# cv2:OpenCV库，用于计算机视觉任务
import cv2
# streamlit:一个用于构建数据应用程序的框架
import streamlit as st

import os
import openai
from dotenv import load_dotenv

from tensorflow import keras

# Keras是TensorFlow的高级API，用于构建和训练神经网络模型
# model_from_json 函数用于从JSON格式的字符串中加载Keras模型。
from keras.models import model_from_json
# img_to_array 函数用于将图像对象(例如PIL图像)转换为NumPy数组。
# NumPy数组是深度学习模型通常需要的输入格式
from keras.preprocessing.image import img_to_array

# streamlit_webrtc 一个Streamlit的扩展库，用于处理实时WebRTC视频流
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# 使用一个字典将情绪分类的索引映射到对应的情绪标签
emotion_dict = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'surprise'}

# 加载 .env 文件中的环境变量
load_dotenv()

# 获取环境变量中的 API 密钥
api_key = os.getenv("OPENAI_API_KEY")

# 初始化 OpenAI 客户端
client = openai(api_key=api_key)


def get_comforting_message(emotion):
    prompt = f"请给我一段安慰{emotion}情绪的句子，使用中文回答。"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个亲人，专门安慰人的情绪。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = response.choices[0].message['content'].strip()
    return message


# 打开存储模型架构的JSON文件
# 使用 with 语句自动管理文件的打开和关闭
with open('emotion_model1.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# json_file = open('emotion_model1.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()

# 从JSON内容加载模型架构
classifier = model_from_json(loaded_model_json)
# 加载模型的权重
classifier.load_weights("emotion_model1.h5")

try:
    # 尝试加载用于面部检测的Haar级联分类器
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

# 创建一个RTCConfiguration对象，指定用于WebRTC连接的STUN服务器
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


# 定义一个继承自VideoTransformerBase的类，用于处理视频帧
class Faceemotion(VideoTransformerBase):

    def __int__(self):
        self.buffer = []  # 用于存储过去几帧的情绪预测结果
        self.buffer_size = 30  # 缓冲区大小，表示需要捕捉30帧
        self.final_emotion = "neutral"
        self.final_message = ""

    # 定义一个方法处理每一帧视频，self是类的实例，frame是当前视频帧
    def transform(self, frame):
        # 将视频帧转换为NumPy数组格式(BGR，每个像素包含蓝、绿、红三种颜色)。NumPy数组是OpenCV处理图像的标准格式。
        img = frame.to_ndarray(format="bgr24")

        # 将图像转换为灰度图，因为面部检测和虚脱图像处理算法在灰度图上运行更快且效果更好，减少了处理的数据量(仅一个通道而不是三个)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 基础窗口尺寸：分类器检测的最小窗口尺寸，通常为24*24像素
        # 当图像缩小后的尺寸小于基础窗口尺寸（如 24x24 像素）时，检测过程停止
        # 如果 minSize 和 maxSize 未显式设置，则 minSize 默认为 (0, 0)，maxSize 默认为图像尺寸

        # 检测图像中的面部，使用Haar级联分类器检测灰度图像中的面部
        # 通过 scaleFactor 参数控制，每次缩小比例为 1/scaleFactor
        # 通过 minNeighbors 参数控制，指定每个候选矩形至少需要的邻近检测框数
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)

        # 遍历检测到的面部
        # `faces`是一个包含每个面部的边界框信息的数组
        # 每个边界框用四个值表示:x(左上角x坐标),y(左上角y坐标),w(宽度),h(高度)
        for (x, y, w, h) in faces:

            # 在检测到的面部周围绘制矩形
            # 使用OpenCV在检测到的面部周围绘制一个矩形。
            # pt1是矩形的左上角坐标，pt2是右下角坐标，coloer指定矩形颜色(红色)，thickness指定矩形的线条粗细
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)

            # 提取面部区域
            # 从灰度图像中提取出检测到的面部区域(ROI)，该区域用于进一步处理
            roi_gray = img_gray[y:y + h, x:x + w]

            # 将面部区域调整为48*48大小，这是情绪识别模型的输入大小
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            # 如果ROI不为空，则进行情绪预测。
            # 检查ROI是否不为空，即像素值总和不为0。确保处理的区域是有效的面部图像，而不是空白区域
            if np.sum([roi_gray]) != 0:
                # 将ROI的像素值从0-255的范围归一化到0-1的范围，并转换为浮点数
                # 标准化有助于提高模型的预测性能
                roi = roi_gray.astype('float') / 255.0

                # 使用Keras的 img_to_array 函数将灰度图像转换为NumPy数组，以符合模型输入格式
                roi = img_to_array(roi)

                # 在数组的第0轴（第一个维度）上增加一个维度，使其形状变为(1, 48, 48, 1)，即一个样本
                # 这是因为神经网络模型通常期望输入是四维张量(批次大小，高度，宽度，通道数)
                roi = np.expand_dims(roi, axis=0)

                # 使用预训练的情绪分类器对ROI进行预测。predict方法返回一个包含每个情绪类别预测值的数组
                # [0]表示我们只关心第一个（也是唯一一个）样本的预测结果
                prediction = classifier.predict(roi)[0]

                # 找出预测值数组中最大值的索引，即模型认为最可能的情绪类别的索引
                maxindex = int(np.argmax(prediction))

                # 使用最大值索引从情绪字典中获取对应的情绪标签
                finalout = emotion_dict[maxindex]

                self.buffer.append(finalout)
                if len(self.buffer) > self.buffer_size:
                    self.buffer.pop(0)
                if len(self.buffer) == self.buffer_size:
                    self.final_emotion = max(set(self.buffer), key=self.buffer.count)
                    self.final_message = get_comforting_message(self.final_emotion)

                # 将情绪标签转换为字符串，方便在图像上绘制
                # output = str(finalout)

            # 在面部区域显示预测的情绪标签
            label_position = (x, y)

            # 在检测到的面部区域上方绘制预测的情绪标签
            # label_position是文本位置
            # cv2.FONT_HERSHEY_SIMPLEX是字体类型
            # 1是字体大小，(0, 255, 0)是文本颜色(绿色)
            # 2是文本线条粗细。
            cv2.putText(img, self.final_emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, self.final_message, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 获取并显示安慰语句
            # message = get_comforting_message(output)
            # cv2.putText(frame, message, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 返回带有绘制矩形框和情绪标签的图像帧，这一帧将显示在WebRTC视频流中
        return img


# 主函数
def main():
    # 设置应用的标题
    st.title("实时人脸情绪检测应用")
    # 包含三个选项的列表，用于导航不同的应用功能
    activities = ["首页", "网络摄像头人脸检测", "关于"]
    # 通过侧边栏的下拉选择框让用户选择活动
    choice = st.sidebar.selectbox("选择标签", activities)

    # Display developer information in the sidebar
    st.sidebar.markdown("""Email: blx030100@gmail.com""")

    # Define different functionalities based on the user's choice
    if choice == "首页":
        # Display information about the application
        st.write("""
                 该应用程序有两个功能
                 1. 使用网络摄像头进行实时人脸检测
                 2. 实时人脸情绪识别
                 """)

    elif choice == "网络摄像头人脸检测":

        # 设置页面标题
        st.header("网络摄像头直播")
        st.write("单击开始使用网络摄像头并检测您的面部情绪")

        # 启动 WebRTC 流媒体，用于实时视频传输和处理
        # key=example:WebRTC组件的键值
        # mode=WebRtcMode.SENDRECV:设置WebRTC模式为发送和接收
        # rtc_configuration=RTC_CONFIGURATION：WebRTC 配置
        # video_processor_factory=Faceemotion：指定视频处理器工厂，用于处理视频流（包括面部检测和情绪识别）
        ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                              video_processor_factory=Faceemotion)

        if ctx.video_processor:
            st.write("检测到的情绪：", ctx.video_processor.final_emotion)
            st.write("安慰话语：", ctx.video_processor.final_message)

    elif choice == "关于":

        # 设置子标题
        st.subheader("关于这个应用程序")

        # 使用 Markdown 格式显示应用和开发者的信息
        st.markdown("""
                    使用 OpenCV、自定义训练的 CNN 模型和 Streamlit 的实时面部情绪检测应用程序。
                    由 xxx 使用 Streamlit Framework、OpenCV、Tensorflow 和 Keras 库开发用于演示目的。
                    如果您有任何建议或想发表评论，请发送电子邮件至 blx030100@gmail.com。
                    感谢造访！
                    """)
    else:
        pass


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
