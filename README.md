# OCR Server

FastAPI-based OCR service for extracting text from PDFs and images. The app includes a web UI for uploading files, a JSON API for OCR jobs, and a DOCX export endpoint for the extracted text.

## Features

- Upload PDF, JPG, or PNG files for OCR.
- Run text detection and recognition with the bundled OpenVINO models.
- View extracted text in the browser.
- Download the result as a DOCX file.
- Check service health with a lightweight API endpoint.

## Requirements

- Python 3.14 or newer.
- The model files shipped in `backend/openvino/`.
- System packages needed by OpenCV, OpenVINO, and PDF/image processing libraries.

## Installation

Using the project dependencies in `pyproject.toml`:

```bash
python -m pip install -U pip
python -m pip install -e .
```

If you prefer the legacy requirements file, install from `requirements.txt` instead.

## Run The App

Start the server from the project root:

```bash
python -m uvicorn main:app --reload
```

Open the app in your browser at `http://127.0.0.1:8000`.

## API Endpoints

- `GET /` - Web UI.
- `GET /api/health` - Health check, returns `{ "status": "healthy" }`.
- `POST /api/detect` - Upload a PDF or image and receive OCR output as JSON.
- `POST /api/download-docx` - Send extracted text and download a formatted DOCX file.

### `POST /api/detect`

Accepts multipart form data with a single `file` field. Supported content types:

- `application/pdf`
- `image/jpeg`
- `image/png`

Example:

```bash
curl -X POST "http://127.0.0.1:8000/api/detect" \
	-F "file=@sample.pdf"
```

## Project Layout

- `main.py` - FastAPI application, routes, and DOCX generation.
- `backend/` - OCR pipeline implementation and model wrappers.
- `backend/openvino/` - Text detection and recognition models used by the server.
- `templates/` - HTML template for the web UI.
- `static/` - Frontend CSS and JavaScript.
- `fonts/` - Local font assets used by the UI.

## Notes

- The server creates temporary job directories under `jobs/` while processing uploads and removes them after each request.
- If OCR startup fails, confirm the model paths in `backend/openvino/` still match the paths referenced in `main.py`.
