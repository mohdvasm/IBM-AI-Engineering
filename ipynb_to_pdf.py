"""Converts ipynb into PDF
"""
import os
import subprocess
import argparse


def convert_notebooks_to_pdf(root_dir):
    """Converts ipynb file to pdf

    Args:
        root_dir (str): root path
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".ipynb") and not filename.startswith("."):
                notebook_path = os.path.join(dirpath, filename)
                base_name = os.path.splitext(filename)[0]

                # Create sibling folder: pdf-version/
                pdf_output_dir = os.path.join(dirpath, "pdf-version")
                os.makedirs(pdf_output_dir, exist_ok=True)

                pdf_path = os.path.join(pdf_output_dir, f"{base_name}.pdf")

                # Remove old PDF if exists
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                    print(f"🗑️ Removed old: {pdf_path}")

                # Convert to PDF
                print(f"📄 Converting to PDF: {notebook_path}")
                try:
                    subprocess.run(
                        [
                            "jupyter",
                            "nbconvert",
                            "--to",
                            "pdf",
                            "--output",
                            f"{base_name}",
                            "--output-dir",
                            pdf_output_dir,
                            notebook_path,
                        ],
                        check=True,
                    )
                    print(f"✅ PDF saved: {pdf_path}\n")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error converting {notebook_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .ipynb to .pdf into sibling pdf-version folders."
    )
    parser.add_argument(
        "folder", type=str, help="Root folder to scan for .ipynb files."
    )
    args = parser.parse_args()

    convert_notebooks_to_pdf(args.folder)
