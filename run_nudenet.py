from nudenet import NudeDetector

# 初始化检测器
nude_detector = NudeDetector()

# 检测图像中的裸露内容
detections = nude_detector.detect('image.jpg')
print(detections)

# 审查图像并输出审查后的图像路径
output_path = nude_detector.censor('image.jpg', output_path='censored_image.jpg')
print(f'Censored image saved at: {output_path}')
