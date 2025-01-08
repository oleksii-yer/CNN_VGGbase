import streamlit as st
from PIL import Image, ImageOps
import io
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

st.title("Класифікація зображення нейромережами")

# Завантадення моделей
try:
    # Через завеликий розмір модель не завантажена. Однак все працювало як треба при запуску.
    VGG_base = load_model('VGG_base.h5')
    own_model = load_model('own_model.h5')
    st.success("Моделі успішно завантажені!")
except Exception as e:
    st.error(f"Помилка під час завантаження моделей: {e}")

# Завантаження та підготовка тренувального датасету для подальших дій
(_, _), (x_test, y_test) = fashion_mnist.load_data()
x_test = x_test / 255
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
x_test = x_test.reshape((10000, 28, 28, 1))

# Меню з різними сторінками
selected = option_menu(menu_title=None, options=["Власна модель", "VGG16 base", "Класифікувати зображення"],
                       menu_icon="cast", default_index=0,
                       orientation="horizontal")

if selected == 'Власна модель':
    st.title('Власна модель')

    # Зображення зі статистикою тренування
    img1 = Image.open('own_model_img1.png')
    img2 = Image.open('own_model_img2.png')

    st.image(img1, use_container_width=True)
    st.image(img2, use_container_width=True)

    y_pred_prob = own_model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Виведення Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.subheader('Загальна оцінка')

    y_test = to_categorical(y_test)
    # Оцінка моделі на тестових даних
    loss, accuracy = own_model.evaluate(x_test, y_test)

    st.text(f"Loss: {loss}")
    st.text(f"Accuracy: {accuracy}")

elif selected == 'VGG16 base':
    st.title('VGG16 base')

    x_test = tf.convert_to_tensor(x_test)
    x_test = tf.image.grayscale_to_rgb(x_test)
    x_test = tf.image.resize(x_test, [32, 32])

    img1 = Image.open('VGG_base_img1.png')
    img2 = Image.open('VGG_base_img2.png')

    st.image(img1, use_container_width=True)
    st.image(img2, use_container_width=True)

    y_pred_prob = VGG_base.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.subheader('Загальна оцінка')

    y_test = to_categorical(y_test)
    loss, accuracy = VGG_base.evaluate(x_test, y_test)

    st.text(f"Loss: {loss}")
    st.text(f"Accuracy: {accuracy}")

else:
    st.title('Класифікувати зображення')

    (input_width, input_height) = (28, 28)
    
    # Можливість завантажити власне зображення
    uploaded_file = st.file_uploader("Виберіть зображення...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Вивід оригінального зображення
        st.image(image, caption='Оброблене зображення', use_container_width=True)

        # Перетворення згідно моделі
        image = ImageOps.grayscale(image)

        resized_image = image.resize((input_width, input_height))

        st.image(resized_image, caption='Оброблене зображення', use_container_width=True)

        # Вибір якою з моделей передбачати
        user_choice = st.selectbox('Вибір моделі для передбачення', ['Власна модель', 'VGG_base'])

        img_array = np.array(resized_image)
        img_array = np.expand_dims(img_array, axis=0)


        if user_choice == 'VGG_base':
            # Додаткові перетворення для моделі на основі VGG
            img_array = img_array.reshape((1, 28, 28, 1))
            VGG_input = tf.convert_to_tensor(img_array)
            VGG_input = tf.image.grayscale_to_rgb(VGG_input)
            VGG_input = tf.image.resize(VGG_input, [32, 32])
            y_pred_prob = VGG_base.predict(VGG_input)
        else:
            y_pred_prob = own_model.predict(img_array)
        y_pred = np.argmax(y_pred_prob, axis=1)

        st.subheader('Передбачення')
        st.text(class_names[y_pred[0]])

