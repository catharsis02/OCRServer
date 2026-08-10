import shutil

def cleanup():
    shutil.rmtree("./det_outputs")
    shutil.rmtree("./pdf2image")
    