from flask import Flask, request, render_template
from transformers import pipeline
import fitz  # PyMuPDF for PDF text extraction
import io

app = Flask(__name__)

# Load pre-trained QA model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")


# Extract text from PDF


def extract_text_from_pdf(pdf_file):
    pdf_bytes = pdf_file.read()  # Read the uploaded file
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")  # Open PDF from bytes
    text = ""
    for page in doc:
        text += page.get_text()
    return text


@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        pdf = request.files["pdf_file"]
        question = request.form["question"]

        if pdf and question:
            pdf_text = extract_text_from_pdf(pdf)
            if pdf_text.strip():  # Ensure there's content to search
                answer = qa_pipeline(question=question, context=pdf_text)["answer"]
            else:
                answer = "Could not extract any readable text from the PDF."

    return render_template("index.html", answer=answer)


if __name__ == "__main__":
    app.run(debug=True)
