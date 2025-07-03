---
# ✋ Hand Gesture Recognition with Action Control

This project is a real-time hand gesture recognition system built with Python. It detects your hand using a webcam and recognizes specific gestures to trigger actions – sign for hello, sign for thankyou, or even controlling hardware (e.g., Arduino/ESP32).

---

## 💡 What This Project Does

- Tracks your hand live using your webcam
- Recognizes different hand gestures (like open palm, fist, etc.)
- Triggers actions based on those gestures
- Can be extended to control devices, UI, or automation tasks

---

## 📦 What’s Inside

```

hand-gesture-recognition/
├── main.py                # Main script to run the project
├── hand\_detector.py       # Detects and tracks hand landmarks
├── gesture\_recognizer.py  # Recognizes hand gestures
├── actions.py             # Maps gestures to actions
├── utils.py               # Helper functions
├── requirements.txt       # Python packages needed
└── README.md              # You're reading it!

````

---

## 🛠️ How to Set It Up

1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/hand-gesture-recognition.git
   cd hand-gesture-recognition
````

2. **Install the required libraries**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main file**

   ```bash
   python main.py
   ```

Make sure your webcam is working!


---
````

## 🧠 How It Works

* Uses **MediaPipe** to detect and track your hand landmarks
* Matches hand poses to predefined gestures
* Each gesture is mapped to a specific action (customizable)

---

## ✨ Example Gesture Actions

| Gesture     | Action                     |
| ----------- | -------------------------- |
| Open palm ✋ | Stop all actions           |
| Fist 👊     | Pause media or lock screen |
| Point 👆    | Move cursor / Scroll       |
| Victory ✌️  | Switch tab or window       |

These can be customized in `gesture_recognizer.py` and `actions.py`.

---

## 🧪 Want to Train Your Own Model?

You can also:

* Collect hand gesture images
* Train a deep learning model using TensorFlow
* Replace rule-based logic with your own model inference

---

## 🔧 Future Ideas

* Recognize both hands at once
* Support for dynamic gestures (e.g., swipes)
* Voice + gesture hybrid control
* Integration with smart home or robots

---

## 👨‍💻 Created By

* Abhindra Krishna K M
* Jenifer Maria Joseph

---

## 📜 License

This project is under the [MIT License](LICENSE). You can use it freely with attribution.

---

## 📬 Contact

For any questions or collaboration:

* Email: `abhindrakrishna@example.com`
* Phone: +91-7736671379

---

Thanks for checking this out! Feel free to contribute or fork the project 🌟

```
