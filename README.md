# Samsung-AI-Challenge-IQA

# Train
1. batch_size : 배치사이즈
2. model_name : vgg, google, resnet, efficient
3. epochs : 에포크 설정
4. early_stop : 조기 종료
5. gpu_id : {0 : cpu, 1 : cuda, 2 : mps}
- python main.py --gpu_id 1 --epochs 100 --early_stop 10 --batch_size 64 --model_name resnet

## 파일 구조
- train/trainer : 훈련시키는 파일
- train/train : 훈련 설정 하는 파일
- train/evalidate : 테스트하는 파일
- train/models/encder_googlenet : GoogleNet 디코더
- train/models/encder_resnet : ResNet 디코더
- train/models/encder_Vgg : Vgg 디코더
- train/models/encder_efficient : efficient 디코더
- train/models/seq2seq : encoder - decoder 연결

# Test
- predict.ipynb : pt 변경 후 실행