from pathlib import Path
import pypdfium2 as pdfium
# import time
from joblib import Parallel, delayed


def _render_page(pdf_path, page_index, save_dir,temp_save_dir, file_name):
    # start = time.perf_counter()
    try:
        with pdfium.PdfDocument(pdf_path) as pdf:
            image = pdf[page_index].render(scale=300 / 72).to_pil()
        image.save(save_dir / f"{file_name}_page_{page_index + 1}.png",format="png",compress_level=1)
        image.save(temp_save_dir / f"{file_name}_page_{page_index + 1}.png",format="png",compress_level=1)

        # print(f"{time.perf_counter() - start} seconds to render")
        return True
    except Exception as exc:
        print(f"Page {page_index} failed: {exc}")
        return False


def extract_photos_from_pdf(
    pdf_path: str | Path, 
    save_dir: str | Path, 
    temp_save_dir: str | Path,
    n_jobs: int = 4,
):
    pdf_path = Path(pdf_path)
    save_dir = Path(save_dir)
    temp_save_dir = Path(temp_save_dir)
    temp_save_dir.mkdir(parents=True,exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = pdf_path.stem

    with pdfium.PdfDocument(pdf_path) as pdf:
        num_pages = len(pdf)


    results = Parallel(n_jobs=n_jobs)(
            delayed(_render_page)(pdf_path, i, save_dir,temp_save_dir, file_name)
            for i in range(num_pages)
        )

    ok = sum(results)
    # print(f"{pdf_path.stem}:{ok/num_pages} pages rendered")
    return results

def clean_image_cache(save_dir: str | Path):
    import shutil
    save_dir = Path(save_dir)
    if save_dir.exists():
        shutil.rmtree(save_dir)


if __name__ == "__main__":
    extract_photos_from_pdf(
        r"C:\Users\kanishkak3\Desktop\onnx_ocr\client_data\IN-KANISHKAK3.pdf",
        save_dir="./output_pages",temp_save_dir="./temp"
    )
