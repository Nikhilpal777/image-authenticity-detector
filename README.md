# 🖼️ Image Authenticity Detector

A deep learning-based web application that detects whether an image is **Real or AI-generated (Fake)** using a trained CNN model. The system also provides visual insights using Grad-CAM.

---

## 🚀 Features

* 📤 Upload an image
* 🤖 Classifies image as **Real or Fake**
* 🔥 Grad-CAM visualization (highlights important regions)
* 📊 Displays prediction confidence
* 🌐 Accessible via web using Flask + Ngrok

---

## 🛠️ Tech Stack

* Python
* Flask (Web Framework)
* TensorFlow / Keras
* OpenCV
* NumPy
* HTML / CSS

---

## 📂 Project Structure

```
ty_project/
│── app.py
│── gradcam.py
│── real_image_authenticity_detector.py
│── image_authenticity_model.keras
│── templates/
│    └── index.html
│── static/
```

---

## ⚙️ How to Run Locally

1. Clone the repository:

```
git clone https://github.com/Nikhilpal777/image-authenticity-detector.git
```

2. Navigate to project folder:

```
cd image-authenticity-detector
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the Flask app:

```
python app.py
```

5. Open in browser:

```
http://127.0.0.1:5000
```

---

## 🌍 Live Demo (Using Ngrok)

1. Start Flask app:

```
python app.py
```

2. Run Ngrok:

```
ngrok http 5000
```

3. Use the generated public URL:

```
https://dour-nonsufferably-jesica.ngrok-free.dev

---

## 📌 Future Improvements

* Add database to store uploaded images
* Improve model accuracy with larger dataset
* Enhance UI/UX design
* Deploy on cloud platforms (AWS / Render)

---

## 👨‍💻 Author

**Nikhil Pal**
