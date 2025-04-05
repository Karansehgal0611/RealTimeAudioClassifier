🚨 Real-Time Emergency Sound Classifier

A real-time audio classification system that detects emergency sounds (such as sirens and car horns) using deep learning and live microphone input.

🔑 Features
🎤 Real-Time Audio Capture — Continuously processes live microphone input

🚨 Emergency Sound Detection — Classifies sirens, car horns, etc.

📊 Live Visualization — Displays spectrogram and classification results in real-time

⚙️ Configurable Thresholds — Adjust detection sensitivity as needed

⚡ Lightweight Model — Runs smoothly on standard hardware with TensorFlow

🧰 Prerequisites
Python 3.7+

A working microphone (built-in or external)

🔧 Installation
Clone this repository:

bash
Copy
Edit
git clone https://github.com/yourusername/SoundClassifier.git
cd SoundClassifier
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
▶️ Usage
Run the real-time classifier:

bash
Copy
Edit
python realtime_classifier.py
🕹 Controls
Press Ctrl + C to stop the program

Detected emergency sounds will be highlighted in red

⚙️ Configuration
You can tweak the following parameters in realtime_classifier.py:

Parameter	Description	Default
window_size	Duration of each audio window (in seconds)	1.0
threshold	Confidence threshold for alerts	0.7
sample_rate	Audio sampling rate	22050 Hz
🧠 Model Details
Architecture: Convolutional Neural Network (CNN)

Input Format: 64×64 Mel-spectrograms

Training Dataset: UrbanSound8K (sirens, car horns, etc.)

Accuracy: ~92% on test set

Model File: emergency_sound_model.h5

(Note: The model file is not included if it's over 100MB—please download separately.)

📁 Project Structure
bash
Copy
Edit
SoundClassifier/
├── realtime_classifier.py       # Main application script
├── emergency_sound_model.h5     # Pre-trained TensorFlow model
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md                    # Project documentation
🛠 Troubleshooting
❌ Microphone Not Detected
Make sure microphone permissions are granted to your terminal or IDE

Ensure the correct input device is selected in your system settings

❌ Model Not Loading
Verify that emergency_sound_model.h5 is in the project root

Check for TensorFlow version compatibility with your system and Python version

🌟 Future Improvements
Expand to support additional emergency sound types

Implement cloud-based inference

Develop a desktop/mobile app interface

📝 License
This project is licensed under the MIT License.

📬 Contact
For issues, suggestions, or contributions, feel free to:

Open a GitHub Issue

Contact me at: karansehgal0310@gmail.com

🚑 Disclaimer: This is a prototype built for educational and experimental purposes only. It is not intended for deployment in real-world emergency system
