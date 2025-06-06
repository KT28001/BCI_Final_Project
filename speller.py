import tkinter as tk
from tkinter import filedialog
import numpy as np
import time
import pickle
from sklearn.svm import SVC
from scipy.signal import resample
import fbcca
from scipy.io import loadmat
import time

KEYS = [
    'A', 'B', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X',
    'Y', 'Z', '0', '1', '2', '3',
    '4', '5', '6', '7', '8', '9'
]

ROW_FREQUENCIES = [8, 9, 10, 11, 12, 13] 
col_freqs = [8, 9, 10, 11, 12, 13] 

class SSVEPSpellerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SSVEP Speller")
        self.frame = tk.Frame(self.root)
        self.frame.pack()

        self.phase = "row"
        self.selected_row = None
        self.selected_char = None
        self.buttons = []
        self.start_time = time.time()
        self.running = True
        self.typed_text = ""

        self.output_label = tk.Label(self.root, text="", font=("Courier", 16), anchor="w")
        self.output_label.pack(fill="x", padx=10, pady=5)

        self.create_keyboard()
        self.root.after(50, self.update_blinking)

        self.status = tk.Label(self.root, text="Press Ctrl+O to drop file and classify SSVEP.")
        self.status.pack()

        self.root.bind("<Control-o>", self.drop_file_event)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_keyboard(self):
        for i in range(6):
            row_buttons = []
            for j in range(6):
                key = KEYS[i * 6 + j]
                btn = tk.Button(self.frame, text=key, width=5, height=2, bg="white")
                btn.grid(row=i, column=j, padx=5, pady=5)
                row_buttons.append(btn)
            self.buttons.append(row_buttons)

    def update_blinking(self):
        if not self.running:
            return

        now = time.time() - self.start_time

        if self.phase == "row":
            for row in range(6):
                freq = ROW_FREQUENCIES[row]
                on = int(now * freq) % 2 == 0
                color = "black" if on else "white"
                for btn in self.buttons[row]:
                    btn.configure(bg=color)
        elif self.phase == "char" and self.selected_row is not None:
            for col in range(6):
                freq = 6 + col
                on = int(now * freq) % 2 == 0
                color = "black" if on else "white"
                self.buttons[self.selected_row][col].configure(bg=color)
        else:
            for row in range(6):
                for btn in self.buttons[row]:
                    btn.configure(bg="white")

        self.root.after(50, self.update_blinking)

    def drop_file_event(self, event=None):
        file_path = filedialog.askopenfilename(title="Select EEG File")
        if not file_path:
            return
        freq = classify_freq_from_file(file_path)
        print(freq)
        self.handle_classification(freq)

    def handle_classification(self, freq):
        if self.phase == "row":
            if freq in ROW_FREQUENCIES:
                self.selected_row = ROW_FREQUENCIES.index(freq)
                self.phase = "char"
                self.status.config(text=f"Row {self.selected_row + 1} selected. Drop again to choose character.")
            else:
                self.status.config(text=f"Unrecognized row frequency: {freq} Hz.")
        elif self.phase == "char":
            if freq in col_freqs:
                selected_col = col_freqs.index(freq)
                self.selected_char = KEYS[self.selected_row * 6 + selected_col]

                self.typed_text += self.selected_char
                self.output_label.config(text=self.typed_text)

                self.status.config(text=f"Character selected: {self.selected_char}")
                self.phase = "row"
                self.selected_row = None
            else:
                self.status.config(text=f"Unrecognized char frequency: {freq} Hz.")

    def on_closing(self):
        self.running = False
        self.root.destroy()


def classify_freq_from_file(file_path):
    with open('model.pkl', 'rb') as f:
        clf2 = pickle.load(f)
        mat_data = loadmat(file_path)
        raw_data = mat_data['extracted_data']
        
        fs_original = 1000
        fs_target = 250
        samples_target = int(raw_data.shape[1] * fs_target / fs_original)
        X_down = resample(raw_data, samples_target, axis=1)

        selected_channels = [48, 54, 55, 56, 57, 58, 61, 62, 63] 
        X_down = X_down[selected_channels, :]
        # print(X_down.shape)
        freqs = np.arange(8, 15)
        epoch = fbcca.fbcca_full(X_down, freqs, fs_target)
    return clf2.predict(epoch.reshape(1, -1))

if __name__ == "__main__":
    root = tk.Tk()
    app = SSVEPSpellerGUI(root)
    root.mainloop()
