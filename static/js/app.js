const dropzone      = document.getElementById('dropzone');
const fileInput     = document.getElementById('file-input');
const dropzoneEmpty = document.getElementById('dropzone-empty');
const previewWrap   = document.getElementById('preview-wrap');
const previewImg    = document.getElementById('preview-img');
const pdfInfoWrap   = document.getElementById('pdf-info-wrap');
const pdfFileName   = document.getElementById('pdf-file-name');
const pdfFileSize   = document.getElementById('pdf-file-size');
const runBtn        = document.getElementById('run-btn');
const errorMsg      = document.getElementById('error-msg');
const textOutput    = document.getElementById('text-output');
const textOutputWrap= document.getElementById('text-output-wrap');
const emptyState    = document.getElementById('empty-state');
const resultCount   = document.getElementById('result-count');
const actionBar     = document.getElementById('action-bar');
const copyBtn       = document.getElementById('copy-btn');
const downloadBtn   = document.getElementById('download-btn');
const fileInfo      = document.getElementById('file-info');
const fileName      = document.getElementById('file-name');
const clearBtn      = document.getElementById('clear-btn');

let currentFile   = null;
let extractedText = '';
let lastJobId     = '';

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.hidden = false;
}

function clearError() {
  errorMsg.hidden = true;
  errorMsg.textContent = '';
}

function resetAll() {
  currentFile = null;
  extractedText = '';
  lastJobId = '';

  previewWrap.hidden = true;
  pdfInfoWrap.hidden = true;
  dropzoneEmpty.hidden = false;
  dropzone.classList.remove('has-file');

  fileInput.value = '';
  fileInfo.hidden = true;
  fileName.textContent = '';

  emptyState.hidden = false;
  textOutput.hidden = true;
  textOutput.textContent = '';
  actionBar.hidden = true;
  resultCount.textContent = '0 lines';

  runBtn.disabled = true;
  runBtn.textContent = 'Run recognition';
  clearError();
}

function setFile(file) {
  if (!file) return;

  if (!file.type.startsWith('image/') && file.type !== 'application/pdf') {
    showError("Unsupported file type. Use PDF, JPG, or PNG.");
    return;
  }

  clearError();
  currentFile = file;

  fileName.textContent = file.name;
  fileInfo.hidden = false;

  emptyState.hidden = false;
  textOutput.hidden = true;
  textOutput.textContent = '';
  actionBar.hidden = true;
  resultCount.textContent = '0 lines';
  extractedText = '';

  dropzone.classList.add('has-file');

  if (file.type === 'application/pdf') {

    previewWrap.hidden = true;
    pdfInfoWrap.hidden = false;
    dropzoneEmpty.hidden = true;
    pdfFileName.textContent = file.name;
    pdfFileSize.textContent = formatBytes(file.size);
  } else {

    pdfInfoWrap.hidden = true;
    previewWrap.hidden = false;
    dropzoneEmpty.hidden = true;
    previewImg.src = URL.createObjectURL(file);
  }

  runBtn.disabled = false;
}

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragging');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragging'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragging');
  setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => setFile(fileInput.files[0]));

clearBtn.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  resetAll();
});

runBtn.addEventListener('click', async () => {
  if (!currentFile) return;
  clearError();
  runBtn.disabled = true;
  runBtn.textContent = 'Processing…';

  previewWrap.classList.add('scanning');
  pdfInfoWrap.classList.add('scanning');

  const formData = new FormData();
  formData.append('file', currentFile);

  try {
    const res = await fetch('/api/detect', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status}).`);
    }
    const data = await res.json();
    displayResults(data);
  } catch (err) {
    showError(err.message || 'Something went wrong processing that file.');
  } finally {
    previewWrap.classList.remove('scanning');
    pdfInfoWrap.classList.remove('scanning');
    runBtn.disabled = false;
    runBtn.textContent = 'Run recognition';
  }
});

function displayResults(data) {
  const lines = data.final_texts || [];
  lastJobId = data.job_id || '';

  const cleanLines = lines.filter(line => !line.startsWith('===='));
  extractedText = cleanLines.join('\n');

  const lineCount = cleanLines.filter(l => l.trim().length > 0).length;
  resultCount.textContent = `${lineCount} line${lineCount === 1 ? '' : 's'}`;

  emptyState.hidden = true;
  textOutput.hidden = false;
  textOutput.textContent = extractedText;
  actionBar.hidden = false;

  textOutputWrap.classList.remove('results-appear');
  void textOutputWrap.offsetWidth;
  textOutputWrap.classList.add('results-appear');
  actionBar.classList.remove('results-appear');
  void actionBar.offsetWidth;
  actionBar.classList.add('results-appear');
}

// AI Gen
copyBtn.addEventListener('click', async () => {
  if (!extractedText) return;
  try {
    await navigator.clipboard.writeText(extractedText);
    copyBtn.classList.add('copied');
    const original = copyBtn.innerHTML;
    copyBtn.innerHTML = '<span class="action-icon">✓</span> Copied!';
    setTimeout(() => {
      copyBtn.classList.remove('copied');
      copyBtn.innerHTML = original;
    }, 2000);
  } catch {
    showError('Failed to copy text to clipboard.');
  }
});

// AI Gen
downloadBtn.addEventListener('click', async () => {
  if (!extractedText) return;

  downloadBtn.innerHTML = '<span class="action-icon">⏳</span> Generating…';
  downloadBtn.disabled = true;

  try {
    const res = await fetch('/api/download-docx', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: extractedText }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'DOCX generation failed.');
    }

    const blob = await res.blob();
    const docxName = currentFile
      ? currentFile.name.replace(/\.[^/.]+$/, '') + '_ocr.docx'
      : 'ocr_output.docx';

    // Trigger download
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = docxName;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
  } catch (err) {
    showError(err.message || 'Failed to generate DOCX.');
  } finally {
    downloadBtn.innerHTML = '<span class="action-icon">📄</span> Download DOCX';
    downloadBtn.disabled = false;
  }
});
