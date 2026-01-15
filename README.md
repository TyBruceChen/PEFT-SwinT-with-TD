# PEFT-SwinT-with-TD
### Brief Introduction:

Parameter Efficient Fine-tuning on Swin Transformer by tensor decomposition of low rank matrices. This work was based on the research of [CaRA](https://github.com/BonnBytes/CaRA). The weight and bias matrices in attention, FFW, and MLP layers are factorized into low rank matrices through CP decomposition. During training, only these matrices are trainable, while the rest from pretrained Swin Transformer are frozen except the last classfication layer. A simple domain-specific task proves this technology shifting can even achieve higher accuracy while largely reduce trainable parameters.

<img width="1184" height="703" alt="image" src="https://github.com/user-attachments/assets/0a60619b-8571-4aeb-ab9e-5990b1cf6826" />


### Deficits:
Although the trainable parameters were heavily reduced, the training time increased instead. I guess the reason is due to no corresponding optimization algorithm for gradient backpropagation. Also, when I tried to train with other large benchmark classfication dataset, the overfitting is obvious.

### Pyramid Design Architecture:

<img width="1200" height="565" alt="image" src="https://github.com/user-attachments/assets/37726bb2-d34f-42fc-8aaf-af4b427d9f3f" />
