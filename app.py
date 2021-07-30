import torch
import numpy as np
from PIL import Image
import streamlit as st
from torchvision import transforms
from efficientnet_pytorch import EfficientNet


device = 'cpu'
MODEL_PATH = "model.bin"


@st.cache(allow_output_mutation=True)
def load_model():
    model = EfficientNet.from_name('efficientnet-b0', num_classes=199)
    model.load_state_dict(torch.load(f"{MODEL_PATH}"))
    return model


with st.spinner('Model is being loaded..'):
    model = load_model()
    model.to(device)
    model.eval()
    classes = np.load("classes.npy").tolist()

st.write("""
         # ButterFly Classification
         """
         )

st.set_option('deprecation.showfileUploaderEncoding', False)

file = st.file_uploader("", type=["jpg", "png"])

st.set_option('deprecation.showfileUploaderEncoding', False)


def import_and_predict(image_data, model):
    mytransform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    image = mytransform(image_data)
    image = torch.unsqueeze(image, dim=0)
    image = image.to(device)
    with torch.no_grad():
        output = torch.nn.functional.softmax(model(image), dim=1)
    out = output.detach().cpu().numpy().tolist()
    pred = np.argmax(out)
    score = np.max(out)
    prediction = classes[pred]
    return prediction, score


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file).convert("RGB")
    st.image(image, use_column_width=True)
    prediction, score = import_and_predict(image, model)
    st.write(f"Class of ButterFly : {prediction}")
    st.write(f"Confidence Score : {score*100:.2f}%")
