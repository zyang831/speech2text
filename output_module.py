import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton,
    QFileDialog, QSlider, QLabel, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
import spacy
import threading
from input_module import RealTimeASR

class OutputModule(QWidget):
    # Define the signal at class level
    text_received = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spacy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')
            
        self.initUI()
        self.transcribed_text = ""
        self.asr = RealTimeASR(callback=self.handle_text)
        self.asr_thread = threading.Thread(target=self.asr.start_streaming)

        # Connect the signal to the slot
        self.text_received.connect(self.update_text)

    def initUI(self):
        layout = QVBoxLayout()

        # Text display area
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

        # Accessibility features
        font_size_label = QLabel('Font Size:', self)
        layout.addWidget(font_size_label)
        self.font_size_slider = QSlider(Qt.Horizontal, self)
        self.font_size_slider.setMinimum(8)
        self.font_size_slider.setMaximum(32)
        self.font_size_slider.setValue(12)
        self.font_size_slider.valueChanged.connect(self.change_font_size)
        layout.addWidget(self.font_size_slider)

        self.high_contrast_checkbox = QCheckBox('High Contrast Mode', self)
        self.high_contrast_checkbox.stateChanged.connect(self.toggle_high_contrast)
        layout.addWidget(self.high_contrast_checkbox)

        # Export button
        export_button = QPushButton('Export', self)
        export_button.clicked.connect(self.export_text)
        layout.addWidget(export_button)

        self.setLayout(layout)
        self.setWindowTitle('Real-Time Transcription')
        self.show()

    def change_font_size(self, value):
        self.text_edit.setStyleSheet(f"font-size: {value}pt;")

    def toggle_high_contrast(self, state):
        if state == Qt.Checked:
            self.setStyleSheet("background-color: black; color: white;")
        else:
            self.setStyleSheet("")

    @pyqtSlot(str)
    def update_text(self, text):
        # This runs on the main thread
        processed_text = self.process_text(text)
        self.transcribed_text += processed_text + " "
        self.text_edit.setPlainText(self.transcribed_text)

    def handle_text(self, text):
        # Emit the signal instead of directly updating UI
        self.text_received.emit(text)

    def process_text(self, text):
        doc = self.nlp(text)
        sentences = [sent.text.capitalize() for sent in doc.sents]
        return ' '.join(sentences)

    def export_text(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Transcription", "", 
            "Word Documents (*.docx);;Markdown Files (*.md);;Text Files (*.txt)", 
            options=options)
        if file_name:
            if file_name.endswith('.docx'):
                self.save_docx(file_name)
            elif file_name.endswith('.md'):
                self.save_md(file_name)
            else:
                self.save_txt(file_name)

    def save_docx(self, file_name):
        from docx import Document
        document = Document()
        document.add_paragraph(self.transcribed_text)
        document.save(file_name)

    def save_md(self, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(self.transcribed_text)

    def save_txt(self, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(self.transcribed_text)

    def start(self):
        self.asr_thread.start()

    def closeEvent(self, event):
        self.asr.is_running = False
        self.asr_thread.join()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    output_module = OutputModule()
    output_module.start()
    sys.exit(app.exec_())
